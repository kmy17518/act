"""uint8 cache of already-resized RGB frames for random-access ACT training.

Random access into the packed HEVC training videos (GOP 8, no B-frames) decodes about 4.5 frames per
requested frame. With three cameras that is ~25 ms of single-core decoding plus ~8 ms of antialiased
resizing per training sample, which caps a 24-worker loader near 900 samples/s and leaves the GPU
waiting. Decoding every selected video once, sequentially, and storing the *resized* frames as uint8
turns a sample into three ~170 KB memcpys from a memory-mapped file.

Cached pixels are ``preprocess_image(frame, image_size)`` -- the exact training/serving resize -- rounded
to the nearest 1/255 (``quantize_image``). The training tensor rebuilt as ``cache.float() / 255`` therefore
differs from the native float path by at most 0.5/255 per value, below the 8-bit quantization of the
lossy source video itself. With no B-frames, decoding from the start of the stream yields the same
pictures as the native seek-then-decode reader; ``verify_frame_cache`` checks this bit for bit on random
samples against ``B1KDataset.raw_sample``.

Layout mirrors the dataset tree under the cache root (one entry per packed video):

    <cache_root>/videos/<video_key>/chunk-XXX/file-YYY.frames.npy   uint8 [N, H, W, 3] RGB, resized
    <cache_root>/videos/<video_key>/chunk-XXX/file-YYY.pts.npy      float64 [N] presentation seconds
    <cache_root>/videos/<video_key>/chunk-XXX/file-YYY.json         manifest (source size/mtime, size, N)

Entries are written atomically (temporary files, fsync, rename) and validated against the source
file's size and mtime, so a stale or partially built entry is rebuilt rather than trusted.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import multiprocessing
import os
from pathlib import Path
import time

import av
import numpy as np
import torch

from b1k_dataset import B1KDataset, VIDEO_KEYS, preprocess_image

FORMAT = 'act_b1k_frame_cache_v1'
LOGGER = logging.getLogger(__name__)


def quantize_image(tensor):
    """Float CHW image in [0, 1] (as `preprocess_image` returns) to the nearest uint8 HWC RGB array."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError('Expected a float CHW RGB image')
    return torch.round(tensor * 255).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


def dequantize_images(images):
    """Cached uint8 (..., H, W, 3) tensor to the float (..., 3, H, W) tensor `preprocess_image` produces.

    Values are exactly `uint8 / 255` on every device: dividing by a 0-dim tensor runs a true IEEE
    division kernel, whereas a Python scalar divisor becomes a reciprocal multiply on CUDA that is one
    ulp off for some of the 256 values (so this matches `preprocess_image` bit for bit). Elementwise
    ops preserve dense strides, so each image stays in channels-last memory order (the tensor-core
    convolution fast path). A (batch, camera, H, W, 3) batch is stored camera-major first so that every
    `images[:, camera]` is one dense channels-last (batch, 3, H, W) block and convolutions need no
    relayout copy.
    """
    if images.dtype != torch.uint8 or images.shape[-1] != 3:
        raise ValueError('Expected uint8 (..., H, W, 3) frames')
    if images.dim() == 5:
        images = images.transpose(0, 1).contiguous().transpose(0, 1)
    return images.movedim(-1, -3).float().div_(torch.tensor(255.0, dtype=torch.float32, device=images.device))


def cache_stem(cache_root, dataset_root, video_path):
    relative = Path(video_path).resolve().relative_to(Path(dataset_root).resolve())
    return Path(cache_root) / relative.parent / relative.stem


def entry_paths(stem):
    stem = Path(stem)
    return (stem.with_name(stem.name + '.frames.npy'), stem.with_name(stem.name + '.pts.npy'),
            stem.with_name(stem.name + '.json'))


def _manifest_matches(manifest, source, image_size):
    stat = Path(source).stat()
    return (manifest.get('format') == FORMAT and manifest.get('image_size') == list(image_size)
            and manifest.get('source_size') == stat.st_size
            and manifest.get('source_mtime_ns') == stat.st_mtime_ns)


def entry_is_valid(stem, source, image_size):
    frames_path, pts_path, manifest_path = entry_paths(stem)
    if not (frames_path.is_file() and pts_path.is_file() and manifest_path.is_file()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return False
    return _manifest_matches(manifest, source, image_size)


def _fsync_and_close(array):
    array.flush()
    descriptor = os.open(array.filename, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    del array


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_entry(source, stem, image_size, dataset_root, torch_threads=1):
    """Decode one packed video sequentially into <stem>.frames.npy / .pts.npy / .json."""
    source, stem, image_size = Path(source), Path(stem), tuple(image_size)
    torch.set_num_threads(torch_threads)
    av.logging.set_level(None)
    frames_path, pts_path, manifest_path = entry_paths(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    temporary = frames_path.with_name('.' + frames_path.name + '.tmp')
    started = time.monotonic()
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        # Same single-threaded decoder configuration as B1KDataset._decode.
        stream.thread_count = 1
        time_base = float(stream.time_base)
        expected = int(stream.frames or 0)
        frames = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8,
                                           shape=(max(expected, 1), *image_size, 3))
        pts, count, overflow = [], 0, []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            image = quantize_image(preprocess_image(frame.to_ndarray(format='rgb24'), image_size))
            if count < len(frames):
                frames[count] = image
            else:
                overflow.append(image)
            pts.append(frame.pts * time_base)
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError(f'No decodable frames in {source}')
    pts = np.asarray(pts, dtype=np.float64)
    if np.any(np.diff(pts) <= 0):
        # The reader relies on presentation order == decode order (no B-frames).
        temporary.unlink(missing_ok=True)
        raise ValueError(f'Non-monotonic presentation timestamps in {source}; cannot cache safely')
    if count != len(frames):
        # Stream metadata disagreed with the decoded frame count; rewrite with the exact shape.
        exact = frames_path.with_name('.' + frames_path.name + '.exact.tmp')
        resized = np.lib.format.open_memmap(exact, mode='w+', dtype=np.uint8, shape=(count, *image_size, 3))
        resized[:min(count, len(frames))] = frames[:min(count, len(frames))]
        for i, image in enumerate(overflow, len(frames)):
            resized[i] = image
        del frames
        temporary.unlink()
        temporary, frames = exact, resized
    _fsync_and_close(frames)
    with pts_path.with_name('.' + pts_path.name + '.tmp').open('wb') as handle:
        np.save(handle, pts)
        handle.flush()
        os.fsync(handle.fileno())
    stat = source.stat()
    manifest = {
        'format': FORMAT, 'image_size': list(image_size), 'frames': int(count),
        'source': str(source.resolve().relative_to(Path(dataset_root).resolve())),
        'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
        'pixels': 'preprocess_image (bilinear, antialias, align_corners=False) then round(x * 255)',
        'decoder': f'PyAV {av.__version__}, libavcodec {av.library_versions.get("libavcodec")}',
        'build_seconds': round(time.monotonic() - started, 3),
    }
    manifest_path.with_name('.' + manifest_path.name + '.tmp').write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, frames_path)
    os.replace(pts_path.with_name('.' + pts_path.name + '.tmp'), pts_path)
    os.replace(manifest_path.with_name('.' + manifest_path.name + '.tmp'), manifest_path)
    _sync_directory(stem.parent)
    return manifest


class FrameCacheReader:
    """Random access to built entries; memory maps lazily so it pickles into loader workers cheaply."""

    def __init__(self, cache_root, dataset_root, image_size, tolerance=0.008):
        self.cache_root = Path(cache_root).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.image_size = tuple(image_size)
        self.tolerance = tolerance
        self.entries = {}

    def __getstate__(self):
        return dict(self.__dict__, entries={})

    def close(self):
        self.entries.clear()

    def open(self, path):
        path = str(path)
        entry = self.entries.get(path)
        if entry is None:
            frames_path, pts_path, manifest_path = entry_paths(cache_stem(self.cache_root, self.dataset_root, path))
            if not manifest_path.is_file():
                raise FileNotFoundError(f'No frame cache entry for {path}; build it with scripts/b1k/build_frame_cache.py')
            manifest = json.loads(manifest_path.read_text())
            if not _manifest_matches(manifest, path, self.image_size):
                raise ValueError(f'Stale or mismatched frame cache entry {manifest_path}; rebuild it')
            frames = np.load(frames_path, mmap_mode='r')
            pts = np.load(pts_path)
            if (frames.shape != (manifest['frames'], *self.image_size, 3) or frames.dtype != np.uint8
                    or pts.shape != (manifest['frames'],) or np.any(np.diff(pts) <= 0)):
                raise ValueError(f'Corrupt frame cache entry for {path}; rebuild it')
            entry = self.entries[path] = (frames, pts)
        return entry

    def validate(self, video_paths):
        for path in sorted(str(p) for p in video_paths):
            self.open(path)

    def positions(self, path, timestamps):
        """Row of the frame nearest to each timestamp (earlier frame on ties), as `_decode` selects.

        Raises when any nearest frame is farther than the tolerance, exactly like the native reader.
        """
        _, pts = self.open(path)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        upper = np.clip(np.searchsorted(pts, timestamps, side='left'), 0, len(pts) - 1)
        lower = np.clip(upper - 1, 0, len(pts) - 1)
        use_lower = np.abs(pts[lower] - timestamps) <= np.abs(pts[upper] - timestamps)
        chosen = np.where(use_lower, lower, upper)
        distance = np.abs(pts[chosen] - timestamps)
        if np.any(distance > self.tolerance):
            worst = int(np.argmax(distance))
            raise ValueError(f'Video timestamp mismatch in {path}: requested {timestamps[worst]:.6f}, '
                             f'nearest error {distance[worst]:.6f}s (tolerance {self.tolerance})')
        return chosen

    def read(self, path, position, out=None):
        """One cached frame as a uint8 HWC array (copied out of the memory map, into `out` if given)."""
        frames, _ = self.open(path)
        if out is None:
            return np.array(frames[position])
        np.copyto(out, frames[position])
        return out


def selected_videos(dataset):
    """Video files (sorted) that the dataset's selected episodes touch, all cameras."""
    return sorted({dataset.video_path(ep, key) for ep in dataset.episodes for key in VIDEO_KEYS})


def build_frame_cache(dataset, cache_root, workers=None, cpu_budget=None, log=LOGGER.info):
    """Build missing/stale entries for every selected video; returns the list of selected videos."""
    cache_root = Path(cache_root).resolve()
    if cache_root == dataset.root or dataset.root in cache_root.parents:
        raise ValueError('The frame cache must not live inside the read-only dataset tree')
    videos = selected_videos(dataset)
    pending = [path for path in videos
               if not entry_is_valid(cache_stem(cache_root, dataset.root, path), path, dataset.image_size)]
    log(f'{len(videos)} selected videos, {len(pending)} to build, image_size={list(dataset.image_size)}, '
        f'cache={cache_root}')
    if pending:
        cache_root.mkdir(parents=True, exist_ok=True)
        pending.sort(key=lambda path: path.stat().st_size, reverse=True)  # longest first: short pool tail
        workers = max(1, min(workers or os.cpu_count() or 1, len(pending)))
        threads = max(1, (cpu_budget or workers) // workers)
        started = time.monotonic()
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(build_entry, path, cache_stem(cache_root, dataset.root, path),
                                   dataset.image_size, dataset.root, threads): path for path in pending}
            for done, future in enumerate(as_completed(futures), 1):
                manifest = future.result()
                log(f'[{done}/{len(pending)}] {manifest["source"]}: {manifest["frames"]} frames in '
                    f'{manifest["build_seconds"]}s ({time.monotonic() - started:.0f}s elapsed)')
    return videos


def verify_frame_cache(dataset, cache_root, samples=256, seed=0, log=LOGGER.info):
    """Compare random native decodes (resized, quantized) against the cache byte for byte."""
    reader = FrameCacheReader(cache_root, dataset.root, dataset.image_size, dataset.timestamp_tolerance)
    rng = np.random.default_rng(seed)
    compared = 0
    started = time.monotonic()
    try:
        for index in rng.integers(len(dataset), size=samples):
            position = int(np.searchsorted(dataset.ends, index, side='right'))
            ep = dataset.episodes[position]
            frame = int(index - dataset.starts[position])
            images, _, _, _, _ = dataset.raw_sample(ep['episode_index'], frame)
            for image, (path, target) in zip(images, dataset.camera_timestamps(ep, frame)):
                expected = quantize_image(preprocess_image(image, dataset.image_size))
                actual = reader.read(path, int(reader.positions(path, [target])[0]))
                if expected.shape != actual.shape or not np.array_equal(expected, actual):
                    raise ValueError(f'Frame cache mismatch: episode {ep["episode_index"]} frame {frame} {path}')
                compared += 1
    finally:
        reader.close()
    log(f'verified {compared} cached frames from {samples} samples against native decoding in '
        f'{time.monotonic() - started:.1f}s: identical')
    return compared


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--task-names', nargs='+', help='Default: every complete local task')
    p.add_argument('--image-size', type=int, nargs=2, default=[240, 240], metavar=('HEIGHT', 'WIDTH'))
    p.add_argument('--workers', type=int, default=max(1, min(10, os.cpu_count() or 1)))
    p.add_argument('--cpu-budget', type=int, default=30, help='Total cores to spread over workers (resize threads)')
    p.add_argument('--verify', type=int, default=256, help='Random samples to compare against native decoding; 0 skips')
    p.add_argument('--seed', type=int, default=0)
    return p


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = parser().parse_args(argv)
    dataset = B1KDataset(args.dataset_path, args.task_names, image_size=args.image_size)
    try:
        build_frame_cache(dataset, args.cache_dir, workers=args.workers, cpu_budget=args.cpu_budget)
        if args.verify:
            verify_frame_cache(dataset, args.cache_dir, samples=args.verify, seed=args.seed)
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
