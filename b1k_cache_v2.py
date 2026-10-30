"""Sharded JPEG frames shared by ACT and DP; v1 remains a separate format.

Shards contain concatenated independent JPEGs. Each source's int64 index holds
(shard number, offset, byte count); float64 pts and an atomic JSON manifest make
incremental entries independently recoverable. Readers bound both maps and FDs.
"""
from collections import OrderedDict
import io
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image

try:
    from .b1k_data_core import revision, source_matches
except ImportError:
    from b1k_data_core import revision, source_matches

FORMAT = 'act_b1k_frame_cache_v2'


def stem_for(cache, root, source):
    relative = Path(source).absolute().relative_to(Path(root).absolute())
    return Path(cache) / relative.parent / relative.stem


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('w') as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_array(path, array):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with tmp.open('wb') as f:
        np.save(f, array); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def decode_rgb_jpeg(payload, decoder='torchvision'):
    """CPU libjpeg decoding. torchvision avoids PIL's extra image conversion/copy."""
    if decoder == 'torchvision':
        import torch
        from torchvision.io import decode_jpeg
        encoded=torch.from_numpy(np.frombuffer(payload,dtype=np.uint8).copy())
        return decode_jpeg(encoded).permute(1,2,0).numpy()
    if decoder != 'pillow':
        raise ValueError(f'unknown JPEG decoder: {decoder}')
    with Image.open(io.BytesIO(payload)) as image:
        return np.array(image if image.mode == 'RGB' else image.convert('RGB'))


def jpeg_roundtrip(image, quality=95, subsampling=0):
    out = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(out, format='JPEG', quality=quality, subsampling=subsampling)
    with Image.open(io.BytesIO(out.getvalue())) as im:
        return np.array(im.convert('RGB'))


def jpeg_roundtrip_chw(images, quality=95, subsampling=0):
    """Optional serving parity for float [..., C, H, W] tensors after resize."""
    import torch
    if images.shape[-3] != 3:
        raise ValueError('JPEG round trip expects RGB CHW images')
    shape = images.shape
    pixels = (images.detach().float().cpu().reshape(-1,*shape[-3:]).permute(0,2,3,1)
              .mul(255).round().clamp(0,255).to(torch.uint8).numpy())
    decoded = np.stack([jpeg_roundtrip(frame,quality,subsampling) for frame in pixels])
    result = torch.from_numpy(decoded).permute(0,3,1,2).reshape(shape).to(images.device,images.dtype)
    return result / 255.


def entry_valid(stem, source, root, image_size, dataset_revision, quality, subsampling):
    stem = Path(stem)
    try:
        m = json.loads(stem.with_suffix('.json').read_text())
        if not (m.get('format') == FORMAT and m['image_size'] == list(image_size)
                and m['jpeg_quality'] == quality and m['jpeg_subsampling'] == subsampling
                and source_matches(m, source, root, dataset_revision)):
            return False
        for name in ('index_file', 'pts_file'):
            if not (stem.parent / m[name]).is_file():
                return False
        # Verify each persisted extent, including interrupted/truncated shards.
        for shard, end in zip(m['shards'], m['shard_ends']):
            p = stem.parent / shard
            if not p.is_file() or p.stat().st_size < end:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


class ShardWriter:
    def __init__(self, directory, max_bytes):
        if max_bytes < 1024:
            raise ValueError('shard size must be at least 1024 bytes')
        self.directory, self.max_bytes = Path(directory), int(max_bytes)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.files, self.ends, self.handle = [], [], None

    def append(self, payload):
        if len(payload) > self.max_bytes:
            raise ValueError('one JPEG exceeds shard size')
        if self.handle is None or self.handle.tell() + len(payload) > self.max_bytes:
            self.sync()
            if self.handle is not None:
                self.handle.close()
            name = 'shard-' + uuid.uuid4().hex + '.jpeg'
            self.files.append(name); self.ends.append(0)
            self.handle = open(self.directory / name, 'xb')
        offset = self.handle.tell()
        self.handle.write(payload)
        self.ends[-1] = self.handle.tell()
        return len(self.files) - 1, offset, len(payload)

    def sync(self):
        if self.handle:
            self.handle.flush(); os.fsync(self.handle.fileno())

    def close(self):
        self.sync()
        if self.handle:
            self.handle.close(); self.handle = None


def _build_group_unlocked(sources, cache_root, dataset_root, image_size=(224, 224), *,
                dataset_revision=None, quality=95, subsampling=0, shard_bytes=2_000_000_000,
                device='cpu', batch_size=None, decoder_threads=1):
    """One worker per source chunk directory; torch resize batches may use CUDA.

    Never overwrites a shard. Entry manifests become visible only after shard
    prefixes and indices are fsynced. A crash may leave unreferenced new shards;
    readers and restore validation never treat them as completed entries.
    """
    import av
    import torch
    import torch.nn.functional as F
    batch_size = batch_size or (1 if device == 'cpu' else 32)
    if batch_size < 1 or decoder_threads < 1:
        raise ValueError('batch size and decoder threads must be positive')
    if not 1 <= quality <= 100 or subsampling not in (0, 1, 2):
        raise ValueError('invalid JPEG quality or chroma subsampling')
    root, cache = Path(dataset_root).absolute(), Path(cache_root).absolute()
    if cache.resolve() == root.resolve() or root.resolve() in cache.resolve().parents:
        raise ValueError('cache must be outside the dataset')
    if not sources:
        return []
    dataset_revision = revision(root, dataset_revision)
    stems = [stem_for(cache, root, p) for p in sources]
    if len({p.parent for p in stems}) != 1:
        raise ValueError('a build group must share one chunk directory')
    directory = stems[0].parent
    for p in directory.glob('*.json'):
        if json.loads(p.read_text()).get('format') != FORMAT:
            raise ValueError('v2 must use a separate cache directory; v1 is protected')
    torch.set_num_threads(1); av.logging.set_level(av.logging.ERROR)
    writer = ShardWriter(directory, shard_bytes)
    manifests = []
    try:
        for source, stem in zip(sources, stems):
            if entry_valid(stem, source, root, image_size, dataset_revision, quality, subsampling):
                continue
            (directory / '.catalog.json').unlink(missing_ok=True)
            started = time.perf_counter()
            indices, pts, pending, pending_pts = [], [], [], []
            def flush():
                if not pending:
                    return
                batch = torch.from_numpy(np.stack(pending)).to(device).permute(0, 3, 1, 2).float() / 255.
                batch = F.interpolate(batch, size=tuple(image_size), mode='bilinear', align_corners=False, antialias=True)
                batch = (batch * 255).round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for pixels, timestamp in zip(batch, pending_pts):
                    b = io.BytesIO()
                    Image.fromarray(pixels).save(b, format='JPEG', quality=quality, subsampling=subsampling)
                    indices.append(writer.append(b.getvalue())); pts.append(timestamp)
                pending.clear(); pending_pts.clear()
            with av.open(str(source)) as c:
                stream = c.streams.video[0]; stream.thread_count = decoder_threads
                if decoder_threads > 1:
                    stream.thread_type = 'FRAME'
                for frame in c.decode(stream):
                    if frame.pts is None:
                        continue
                    pending.append(frame.to_ndarray(format='rgb24'))
                    pending_pts.append(float(frame.pts * stream.time_base))
                    if len(pending) == batch_size:
                        flush()
                flush()
            if not pts or np.any(np.diff(pts) <= 0):
                raise ValueError(f'empty/nonmonotonic source {source}')
            index = np.asarray(indices, dtype=np.int64)
            # Keep each manifest's shard list local to this video.
            used, inverse = np.unique(index[:, 0], return_inverse=True)
            index[:, 0] = inverse
            writer.sync()
            generation = uuid.uuid4().hex
            index_name, pts_name = f'{stem.name}.{generation}.index.npy', f'{stem.name}.{generation}.pts.npy'
            atomic_array(stem.parent / index_name, index)
            atomic_array(stem.parent / pts_name, np.asarray(pts, dtype=np.float64))
            manifest = {'format': FORMAT, 'index_file': index_name, 'pts_file': pts_name, 'image_size': list(image_size), 'frames': len(pts),
                        'source': str(Path(source).absolute().relative_to(root)), 'source_size': Path(source).stat().st_size,
                        'dataset_revision': dataset_revision, 'jpeg_quality': quality, 'jpeg_subsampling': subsampling,
                        'resize': 'torch bilinear antialias align_corners=False then round(x*255)',
                        'device': device, 'decoder': f'PyAV {av.__version__}',
                        'shards': [writer.files[i] for i in used], 'shard_ends': [writer.ends[i] for i in used],
                        'build_seconds': time.perf_counter() - started, 'jpeg_bytes': int(index[:, 2].sum())}
            atomic_json(stem.with_suffix('.json'), manifest)
            manifests.append(manifest)
    finally:
        writer.close()
    if manifests or not (directory / '.catalog.json').is_file():
        consolidate_chunk(directory)
    return manifests


def build_group(sources, cache_root, dataset_root, image_size=(224, 224), **kwargs):
    """Serialize builders for a chunk; readers continue using published generations."""
    sources=list(sources)
    if not sources:
        return []
    root,cache=Path(dataset_root).absolute(),Path(cache_root).absolute()
    if cache.resolve() == root.resolve() or root.resolve() in cache.resolve().parents:
        raise ValueError('cache must be outside the dataset')
    directories={stem_for(cache,root,p).parent for p in sources}
    if len(directories) != 1:
        raise ValueError('a build group must share one chunk directory')
    directory=next(iter(directories))
    for path in directory.glob('*.json'):
        if json.loads(path.read_text()).get('format') != FORMAT:
            raise ValueError('v2 must use a separate cache directory; v1 is protected')
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.build.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _build_group_unlocked(sources,cache,root,image_size,**kwargs)


def consolidate_chunk(directory):
    """Publish mmap indices and a seekable manifest catalog for a completed chunk.

    Per-entry manifests remain the incremental build journal. The catalog avoids
    opening two .npy files plus JSON for every randomly selected video. Publication
    is atomic; an interrupted consolidation leaves the journal readable.
    """
    directory = Path(directory)
    entries = []
    for path in sorted(directory.glob('*.json')):
        if path.name.startswith('.'):
            continue
        m = json.loads(path.read_text())
        if m.get('format') == FORMAT and 'source' in m:
            entries.append((path.stem, m))
    if not entries:
        return
    generation = 'catalog-' + uuid.uuid4().hex
    files = {k: generation + '.' + k + suffix for k,suffix in
             [('index','.npy'), ('pts','.npy'), ('names','.npy'), ('offsets','.npy'), ('records','.jsonbytes')]}
    total = sum(m['frames'] for _,m in entries)
    idx = np.lib.format.open_memmap(directory/files['index'], mode='w+', dtype=np.int64, shape=(total,3))
    pts = np.lib.format.open_memmap(directory/files['pts'], mode='w+', dtype=np.float64, shape=(total,))
    offsets = [0]; row = 0
    with (directory/files['records']).open('wb') as output:
        for _,manifest in entries:
            m = dict(manifest); end = row+m['frames']
            idx[row:end] = np.load(directory/m['index_file'], mmap_mode='r')
            pts[row:end] = np.load(directory/m['pts_file'], mmap_mode='r')
            m.update(index_file=files['index'], pts_file=files['pts'], row_start=row)
            output.write(json.dumps(m,separators=(',',':')).encode());offsets.append(output.tell());row=end
        output.flush();os.fsync(output.fileno())
    idx.flush();pts.flush();del idx,pts
    atomic_array(directory/files['names'], np.asarray([name for name,_ in entries]))
    atomic_array(directory/files['offsets'], np.asarray(offsets,dtype=np.int64))
    atomic_json(directory/'.catalog.json', {'format':FORMAT, 'kind':'chunk_catalog', 'files':files,
                                           'entries':len(entries),'frames':total})
    return {'entries':len(entries), 'frames':total, 'files':files}


class ShardedReader:
    def __init__(self, cache_root, dataset_root, image_size, tolerance=0.008, *, max_entries=16, max_shards=16):
        self.cache_root, self.dataset_root = Path(cache_root).absolute(), Path(dataset_root).absolute()
        self.image_size, self.tolerance = tuple(image_size), tolerance
        self.dataset_revision = revision(dataset_root)
        self.max_entries, self.max_shards = max_entries, max_shards
        self.decoder=os.environ.get('B1K_JPEG_DECODER','torchvision')
        if self.decoder not in ('torchvision','pillow'):
            raise ValueError('B1K_JPEG_DECODER must be torchvision or pillow')
        self.entries, self.shards, self.catalogs = OrderedDict(), OrderedDict(), OrderedDict()

    def __getstate__(self):
        return dict(self.__dict__, entries=OrderedDict(), shards=OrderedDict(), catalogs=OrderedDict())

    def close(self):
        self.entries.clear()
        self.catalogs.clear()
        for fd in self.shards.values():
            os.close(fd)
        self.shards.clear()

    def _catalog_entry(self, stem):
        directory = stem.parent
        if directory not in self.catalogs:
            marker = directory / '.catalog.json'
            if not marker.is_file():
                return None
            catalog = json.loads(marker.read_text())
            files = catalog['files']
            maps = {k:np.load(directory/files[k],mmap_mode='r') for k in ('index','pts','names','offsets')}
            maps['records'] = np.memmap(directory/files['records'],mode='r',dtype=np.uint8)
            self.catalogs[directory] = maps
            while len(self.catalogs) > self.max_entries:
                self.catalogs.popitem(last=False)
        self.catalogs.move_to_end(directory)
        maps = self.catalogs[directory]
        i = int(np.searchsorted(maps['names'],stem.name))
        if i >= len(maps['names']) or maps['names'][i] != stem.name:
            return None
        lo,hi = maps['offsets'][i:i+2]
        m = json.loads(maps['records'][lo:hi].tobytes())
        start = m['row_start']; end = start+m['frames']
        return m, maps['index'][start:end], maps['pts'][start:end]

    def open(self, path):
        key = str(path)
        if key not in self.entries:
            stem = stem_for(self.cache_root, self.dataset_root, path)
            catalog = self._catalog_entry(stem)
            m = catalog[0] if catalog else json.loads(stem.with_suffix('.json').read_text())
            if (m.get('format') != FORMAT or m['image_size'] != list(self.image_size)
                    or not source_matches(m, path, self.dataset_root, self.dataset_revision)):
                raise ValueError(f'stale/mismatched v2 cache entry: {stem}')
            if catalog:
                _,idx,pts = catalog
            else:
                idx = np.load(stem.parent / m['index_file'], mmap_mode='r')
                pts = np.load(stem.parent / m['pts_file'], mmap_mode='r')
            if idx.shape != (m['frames'], 3) or idx.dtype != np.int64 or pts.shape != (m['frames'],):
                raise ValueError(f'corrupt v2 index: {stem}')
            self.entries[key] = (idx, pts, [stem.parent / p for p in m['shards']])
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        return self.entries[key]

    def positions(self, path, timestamps):
        _, pts, _ = self.open(path)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        upper = np.clip(np.searchsorted(pts, timestamps), 0, len(pts) - 1)
        lower = np.maximum(upper - 1, 0)
        selected = np.where(np.abs(pts[lower] - timestamps) <= np.abs(pts[upper] - timestamps), lower, upper)
        if np.any(np.abs(pts[selected] - timestamps) > self.tolerance):
            raise ValueError(f'video timestamp mismatch in {path}')
        return selected

    def read(self, path, position, out=None):
        idx, _, files = self.open(path)
        if not 0 <= position < len(idx):
            raise IndexError(position)
        shard, offset, size = map(int, idx[position])
        if not 0 <= shard < len(files) or offset < 0 or size < 1:
            raise ValueError('corrupt JPEG index')
        file = files[shard]
        if file not in self.shards:
            self.shards[file] = os.open(file, os.O_RDONLY)
            while len(self.shards) > self.max_shards:
                _, fd = self.shards.popitem(last=False); os.close(fd)
        self.shards.move_to_end(file)
        payload = os.pread(self.shards[file], size, offset)
        if len(payload) != size:
            raise ValueError(f'truncated JPEG shard: {file}')
        result = decode_rgb_jpeg(payload,self.decoder)
        if result.shape != (*self.image_size, 3):
            raise ValueError('JPEG dimensions disagree with manifest')
        if out is None:
            return result
        np.copyto(out, result); return out

    def validate(self, video_paths):
        for path in video_paths:
            self.open(path)
        self.close()
