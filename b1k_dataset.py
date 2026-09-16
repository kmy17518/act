"""Read-only, bounded-memory ACT samples from local LeRobot v3 Parquet/video."""

from collections import OrderedDict
import hashlib
import json
import logging
from pathlib import Path
import time

import av
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F


STATE_INDICES = list(range(3)) + list(range(53, 57)) + list(range(3, 10)) + [24, 25] + list(range(28, 35)) + [49, 50]
CAMERAS = ['zed_link', 'left_realsense_link', 'right_realsense_link']
VIDEO_KEYS = [f'observation.rgb.{camera}_camera_0' for camera in CAMERAS]
OBS_KEYS = [f'robot_r1::robot_r1:{camera}:Camera:0::rgb' for camera in CAMERAS]
COLUMNS = ['index', 'episode_index', 'frame_index', 'timestamp', 'task_index', 'observation.state', 'action']
LOGGER = logging.getLogger(__name__)


def preprocess_image(image, image_size):
    """RGB/RGBA uint8 HWC to RGB float CHW, resized identically in train/serve."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError('Expected uint8 RGB/RGBA image with shape (H, W, 3|4)')
    if not all(image.shape[:2]):
        raise ValueError('Empty image')
    tensor = torch.from_numpy(np.array(image[..., :3], copy=True)).permute(2, 0, 1).float() / 255.0
    if tuple(tensor.shape[-2:]) != tuple(image_size):
        tensor = F.interpolate(tensor[None], size=image_size, mode='bilinear', align_corners=False,
                               antialias=True)[0]
    return tensor


def preprocess_state(state, task_id, stats, task_map):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (61,) or not np.isfinite(state).all():
        raise ValueError('Expected finite 61-dimensional R1Pro state')
    task_ids = sorted(task_map)
    if task_id not in task_map:
        raise ValueError(f'Unseen task_id {task_id}; checkpoint tasks: {task_map}')
    qpos = (state[STATE_INDICES] - np.asarray(stats['qpos_mean'])) / np.asarray(stats['qpos_std'])
    onehot = np.zeros(len(task_ids), dtype=np.float32)
    onehot[task_ids.index(task_id)] = 1
    return torch.from_numpy(np.concatenate([qpos, onehot]).astype(np.float32))


def _matrix(column):
    flat = pc.list_flatten(column).to_numpy(zero_copy_only=False)
    return flat.reshape(len(column), -1).astype(np.float32, copy=False)


class B1KDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, task_names=None, chunk_size=100, image_size=(240, 240),
                 cache_row_groups=2, video_cache_size=3, timestamp_tolerance=0.008, profile_reads=False):
        self.root = Path(dataset_path).resolve()
        self.info = json.loads((self.root / 'meta/info.json').read_text())
        if self.info.get('codebase_version') != 'v3.0':
            raise ValueError('Expected native LeRobot v3.0 metadata')
        for key, shape in [('action', [23]), ('observation.state', [61])]:
            if self.info['features'][key]['shape'] != shape:
                raise ValueError(f'Expected {key} shape {shape}')
        if chunk_size < 1 or min(image_size) < 1 or cache_row_groups < 1 or video_cache_size < 1:
            raise ValueError('Chunk, image, and cache sizes must be positive')
        self.chunk_size = chunk_size
        self.image_size = tuple(image_size)
        self.cache_row_groups = cache_row_groups
        self.video_cache_size = video_cache_size
        self.timestamp_tolerance = timestamp_tolerance
        self.profile_reads = profile_reads
        self.stats = None
        tasks = pq.read_table(self.root / 'meta/tasks.parquet').to_pylist()
        names = {int(row['task_index']): row.get('task', row.get('__index_level_0__')) for row in tasks}
        if any(not isinstance(name, str) for name in names.values()) or len(set(names.values())) != len(names):
            raise ValueError('meta/tasks.parquet must contain unique task names and task_index values')
        if isinstance(task_names, str):
            task_names = [task_names]
        requested = set(task_names or [])
        unknown = requested - set(names.values())
        if unknown:
            raise ValueError(f'Unknown task names {sorted(unknown)}; available metadata tasks: {sorted(names.values())}')
        selected = {i for i, name in names.items() if not requested or name in requested}
        columns = ['episode_index', 'task_index', 'length', 'dataset_from_index', 'dataset_to_index',
                   'data/chunk_index', 'data/file_index']
        columns += [f'videos/{key}/{field}' for key in VIDEO_KEYS
                    for field in ['chunk_index', 'file_index', 'from_timestamp', 'to_timestamp']]
        self.episodes = []
        existence = {}
        skipped = 0
        metadata_files = sorted((self.root / 'meta/episodes').glob('*/*.parquet'))
        if not metadata_files:
            raise FileNotFoundError(f'No meta/episodes/*/*.parquet under {self.root}')
        for path in metadata_files:
            schema = pq.read_schema(path)
            if 'task_index' in schema.names:
                table = pq.read_table(path, columns=columns)
            else:
                table = pq.read_table(path, columns=[key for key in columns if key != 'task_index'] + ['tasks'])
                by_name = {name: i for i, name in names.items()}
                task_ids = []
                for tasks in table['tasks'].to_pylist():
                    if not tasks or len(tasks) != 1 or tasks[0] not in by_name:
                        raise ValueError(f'{path}: each episode needs one known task or an explicit task_index')
                    task_ids.append(by_name[tasks[0]])
                table = table.append_column('task_index', pa.array(task_ids))
                table = table.select(columns)
            table = table.filter(pc.is_in(table['task_index'], value_set=pa.array(sorted(selected))))
            for ep in table.to_pylist():
                paths = [self.data_path(ep)] + [self.video_path(ep, key) for key in VIDEO_KEYS]
                missing = []
                for file in paths:
                    if file not in existence:
                        existence[file] = file.is_file()
                    if not existence[file]:
                        missing.append(str(file))
                if missing:
                    skipped += 1
                    continue
                if ep['length'] <= 0 or ep['dataset_to_index'] - ep['dataset_from_index'] != ep['length']:
                    raise ValueError(f'Invalid episode bounds: {ep["episode_index"]}')
                self.episodes.append(ep)
        self.episodes.sort(key=lambda ep: ep['episode_index'])
        if not self.episodes:
            raise ValueError(f'No complete local episodes for task names {sorted(requested) or "all"}; '
                             f'{skipped} episodes reference missing Parquet/RGB video files under {self.root}')
        available = {int(ep['task_index']) for ep in self.episodes}
        missing_tasks = requested - {names[i] for i in available}
        if missing_tasks:
            raise ValueError(f'No complete local episodes for requested tasks {sorted(missing_tasks)}; '
                             'check the partial download (Parquet and all three RGB cameras are required)')
        self.task_map = {i: names[i] for i in sorted(available)}
        self.by_id = {int(ep['episode_index']): ep for ep in self.episodes}
        if len(self.by_id) != len(self.episodes):
            raise ValueError('Duplicate episode_index in metadata')
        self.lengths = np.array([ep['length'] for ep in self.episodes], dtype=np.int64)
        self.ends = self.lengths.cumsum()
        self.starts = self.ends - self.lengths
        self._groups = OrderedDict()
        self._footers = OrderedDict()
        self._videos = OrderedDict()
        LOGGER.info('Local dataset: %d episodes, %d frames, %d tasks; skipped %d incomplete episodes',
                    len(self.episodes), len(self), len(self.task_map), skipped)

    def data_path(self, ep):
        return self.root / self.info['data_path'].format(chunk_index=ep['data/chunk_index'],
                                                       file_index=ep['data/file_index'])

    def video_path(self, ep, key):
        return self.root / self.info['video_path'].format(video_key=key,
                         chunk_index=ep[f'videos/{key}/chunk_index'], file_index=ep[f'videos/{key}/file_index'])

    def fingerprint(self):
        files = sorted({self.data_path(ep) for ep in self.episodes})
        payload = {'root': str(self.root), 'episodes': self.episodes,
                   'files': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
                   'state_indices': STATE_INDICES, 'std_correction': 1}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def __len__(self):
        return int(self.ends[-1])

    def _read_rows(self, ep, frame, count):
        path = self.data_path(ep)
        lower = ep['dataset_from_index'] + frame
        upper = min(lower + count, ep['dataset_to_index'])
        if path not in self._footers:
            with pq.ParquetFile(path) as file:
                index_col = file.schema_arrow.names.index('index')
                groups = []
                for i in range(file.num_row_groups):
                    stats = file.metadata.row_group(i).column(index_col).statistics
                    groups.append((i, stats.min if stats and stats.has_min_max else None,
                                   stats.max if stats and stats.has_min_max else None))
            self._footers[path] = groups
            if len(self._footers) > 8:
                self._footers.popitem(last=False)
        self._footers.move_to_end(path)
        tables = []
        for group, first, last in self._footers[path]:
            if first is not None and (last < lower or first >= upper):
                continue
            key = (path, group)
            if key not in self._groups:
                with pq.ParquetFile(path) as file:
                    self._groups[key] = file.read_row_group(group, columns=COLUMNS)
                if len(self._groups) > self.cache_row_groups:
                    self._groups.popitem(last=False)
            self._groups.move_to_end(key)
            table = self._groups[key]
            tables.append(table.filter(pc.and_(pc.greater_equal(table['index'], lower),
                                               pc.less(table['index'], upper))))
        if not tables:
            raise ValueError(f'No rows for episode {ep["episode_index"]} frame {frame} in {path}')
        table = pa.concat_tables(tables).sort_by('index')
        if (table.num_rows != upper - lower or
                table['index'].to_pylist() != list(range(lower, upper)) or
                table['frame_index'].to_pylist() != list(range(frame, frame + upper - lower)) or
                set(table['episode_index'].to_pylist()) != {ep['episode_index']} or
                set(table['task_index'].to_pylist()) != {ep['task_index']}):
            raise ValueError(f'Corrupt episode/frame bounds for episode {ep["episode_index"]} in {path}')
        return table

    def _decode(self, path, timestamp):
        av.logging.set_level(None)
        if path not in self._videos:
            container = av.open(str(path))
            container.streams.video[0].thread_count = 1
            self._videos[path] = container
            if len(self._videos) > self.video_cache_size:
                self._videos.popitem(last=False)[1].close()
        self._videos.move_to_end(path)
        container = self._videos[path]
        stream = container.streams.video[0]
        container.seek(max(0, int(timestamp / float(stream.time_base))), stream=stream, backward=True)
        best = None
        distance = float('inf')
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            time = float(frame.pts * stream.time_base)
            delta = abs(time - timestamp)
            if delta < distance:
                best, distance = frame, delta
            if time >= timestamp:
                break
        if best is None or distance > self.timestamp_tolerance:
            raise ValueError(f'Video timestamp mismatch in {path}: requested {timestamp:.6f}, '
                             f'nearest error {distance:.6f}s (tolerance {self.timestamp_tolerance})')
        return best.to_ndarray(format='rgb24')

    def raw_sample(self, episode_id, frame):
        ep = self.by_id[episode_id]
        if not 0 <= frame < ep['length']:
            raise IndexError(f'Frame {frame} outside episode {episode_id}')
        read_start = time.perf_counter() if self.profile_reads else 0
        table = self._read_rows(ep, frame, self.chunk_size)
        video_start = time.perf_counter() if self.profile_reads else 0
        timestamp = float(table['timestamp'][0].as_py())
        images = []
        for key in VIDEO_KEYS:
            offset = float(ep[f'videos/{key}/from_timestamp'])
            target = offset + timestamp
            if target < offset - self.timestamp_tolerance or target >= ep[f'videos/{key}/to_timestamp']:
                raise ValueError(f'Timestamp {target} outside episode {episode_id} camera {key}')
            images.append(self._decode(self.video_path(ep, key), target))
        if self.profile_reads:
            self._read_timings = {'data/parquet_s': video_start - read_start,
                                  'data/video_decode_s': time.perf_counter() - video_start}
        actions = np.zeros((self.chunk_size, 23), dtype=np.float32)
        actions[:table.num_rows] = _matrix(table['action'])
        is_pad = np.arange(self.chunk_size) >= table.num_rows
        state = np.asarray(table['observation.state'][0].as_py(), dtype=np.float32)
        if not np.isfinite(actions).all():
            raise ValueError('Non-finite actions in dataset')
        return images, state, actions, is_pad, int(ep['task_index'])

    def sample_at(self, episode_id, frame):
        if self.stats is None:
            raise RuntimeError('Set dataset.stats before requesting normalized samples')
        images, state, actions, is_pad, task_id = self.raw_sample(episode_id, frame)
        image = torch.stack([preprocess_image(x, self.image_size) for x in images])
        qpos = preprocess_state(state, task_id, self.stats, self.task_map)
        actions = (actions - np.asarray(self.stats['action_mean'])) / np.asarray(self.stats['action_std'])
        return image, qpos, torch.from_numpy(actions.astype(np.float32)), torch.from_numpy(is_pad)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        begin = time.perf_counter() if self.profile_reads else 0
        pos = int(np.searchsorted(self.ends, index, side='right'))
        sample = self.sample_at(self.episodes[pos]['episode_index'], int(index - self.starts[pos]))
        if self.profile_reads:
            return (*sample, {**self._read_timings, 'data/sample_s': time.perf_counter() - begin})
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_groups=OrderedDict(), _footers=OrderedDict(), _videos=OrderedDict())
        return state

    def close(self):
        for container in self._videos.values():
            container.close()
        self._videos.clear()
        self._groups.clear()
        self._footers.clear()

    def compute_stats(self, max_frames=None):
        """Stream each selected file once; never decode video or retain all frames."""
        count = 0
        mean = np.zeros(48, dtype=np.float64)
        m2 = np.zeros(48, dtype=np.float64)
        by_file = {}
        for ep in self.episodes:
            by_file.setdefault(self.data_path(ep), []).append(ep['episode_index'])
        for path, ids in sorted(by_file.items()):
            with pq.ParquetFile(path) as file:
                for batch in file.iter_batches(batch_size=65536, columns=['episode_index', 'observation.state', 'action']):
                    table = pa.Table.from_batches([batch])
                    table = table.filter(pc.is_in(table['episode_index'], value_set=pa.array(ids)))
                    if max_frames is not None:
                        table = table.slice(0, max(0, max_frames - count))
                    if not table.num_rows:
                        continue
                    x = np.concatenate([_matrix(table['observation.state'])[:, STATE_INDICES],
                                        _matrix(table['action'])], axis=1).astype(np.float64)
                    if not np.isfinite(x).all():
                        raise ValueError(f'Non-finite state/action statistics in {path}')
                    n = len(x)
                    local_mean = x.mean(axis=0)
                    delta = local_mean - mean
                    m2 += ((x - local_mean) ** 2).sum(axis=0) + delta ** 2 * count * n / (count + n)
                    mean += delta * n / (count + n)
                    count += n
                    if max_frames is not None and count >= max_frames:
                        break
            LOGGER.info('Statistics: %d frames after %s', count, path.name)
            if max_frames is not None and count >= max_frames:
                break
        if max_frames is None and count != len(self):
            raise ValueError(f'Full statistics read {count} frames but selected metadata declares {len(self)}')
        if count < 2:
            raise ValueError('At least two frames required for sample-standard-deviation normalization')
        std = np.maximum(np.sqrt(m2 / (count - 1)), 0.01)
        return {'qpos_mean': mean[:25].astype(np.float32).tolist(),
                'qpos_std': std[:25].astype(np.float32).tolist(),
                'action_mean': mean[25:].astype(np.float32).tolist(),
                'action_std': std[25:].astype(np.float32).tolist(),
                'count': count, 'std_correction': 1, 'std_floor': 0.01,
                'approximate': count != len(self), 'fingerprint': self.fingerprint()}


class SplitBatchSampler:
    """Dispatch slices of each optimizer batch to independent loader workers."""
    def __init__(self, sampler, loader_batch_size):
        self.sampler, self.loader_batch_size = sampler, loader_batch_size

    def __iter__(self):
        for batch in self.sampler:
            for start in range(0, len(batch), self.loader_batch_size):
                yield batch[start:start + self.loader_batch_size]

    def __len__(self):
        return len(self.sampler) * ((self.sampler.batch_size + self.loader_batch_size - 1) // self.loader_batch_size)


class StepBatchSampler:
    """Episode-uniform ACT sampling without a full-frame permutation or worker RNG."""
    def __init__(self, dataset, batch_size, start_step, max_steps, seed):
        self.dataset, self.batch_size = dataset, batch_size
        self.start_step, self.max_steps, self.seed = start_step, max_steps, seed

    def __iter__(self):
        for step in range(self.start_step, self.max_steps):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            episodes = rng.integers(len(self.dataset.episodes), size=self.batch_size)
            frames = [int(rng.integers(self.dataset.lengths[ep])) for ep in episodes]
            yield [int(self.dataset.starts[ep]) + frame for ep, frame in zip(episodes, frames)]

    def __len__(self):
        return self.max_steps - self.start_step
