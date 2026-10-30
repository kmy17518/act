"""Portable B1K data primitives. Vendored identically in ACT, DP and openpi.

Version 2: linear exact quantiles (disk-backed), or reproducible Bernoulli
subsampling for quantiles only; all moments include every valid value.
No model imports, source writes, padding inference, or quantile widening.
"""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

VERSION = 2
EPS = 1e-6


def revision(root, explicit=None):
    """Resolve a portable revision; never silently replace identity with mtime.

    HF local-dir download receipts contain the immutable commit on their first
    line. A lowdim-only copy can instead supply B1K_DATASET_REVISION.
    """
    if explicit or os.environ.get('B1K_DATASET_REVISION'):
        return explicit or os.environ['B1K_DATASET_REVISION']
    root = Path(root)
    receipts = root / '.cache/huggingface/download'
    for name in ('meta/info.json.metadata', 'meta/tasks.parquet.metadata'):
        p = receipts / name
        if p.is_file():
            value = p.read_text().splitlines()[0]
            if len(value) == 40 and all(c in '0123456789abcdef' for c in value):
                return value
    # Useful for local/synthetic datasets: hash small metadata plus file sizes.
    # Published datasets should always pass their immutable revision explicitly.
    h = hashlib.sha256()
    for p in sorted((root / 'meta').rglob('*.json')):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    for p in sorted((root / 'meta').rglob('*.parquet')):
        h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return 'metadata-sha256:' + h.hexdigest()


def fingerprint(root, episodes, files, *, dataset_revision=None, **semantics):
    # Preserve the caller's lexical root: datasets and their data/ directories
    # may be symlinks, while file paths are constructed relative to that root.
    root = Path(root).absolute()
    payload = {'version': VERSION, 'revision': revision(root, dataset_revision),
               'episodes': episodes, 'semantics': semantics,
               'files': [(str(Path(p).absolute().relative_to(root)), Path(p).stat().st_size)
                         for p in sorted(set(map(Path, files)))]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_matches(manifest, source, dataset_root, dataset_revision=None):
    """New entries travel across hosts; legacy entries retain their old check."""
    source = Path(source)
    if not source.is_file() or manifest.get('source_size') != source.stat().st_size:
        return False
    if 'dataset_revision' in manifest:
        return manifest['dataset_revision'] == revision(dataset_root, dataset_revision)
    return manifest.get('source_mtime_ns') == source.stat().st_mtime_ns


def valid_chunk_count(lengths, offsets):
    return int(sum(np.maximum(np.asarray(lengths, dtype=np.int64) - abs(int(k)), 0).sum()
                   for k in offsets))


class StreamingStats:
    """Exact moments and disk-backed quantiles with bounded working memory.

    Exact quantiles load/partition one dimension at a time. In auto mode more
    than 500M rows uses a deterministic hash sample (10M rows expected). Sampling
    depends on row ordinal, not update boundaries, and never affects moments.
    Callers must exclude padding BEFORE update. The scratch files are temporary.
    """
    def __init__(self, dimensions, expected_count, *, estimator='auto', sample_size=10_000_000,
                 exact_limit=500_000_000, seed=0, scratch=None, quantiles=True):
        if estimator not in ('auto', 'exact', 'sampled'):
            raise ValueError('estimator must be auto, exact, or sampled')
        if dimensions < 1 or expected_count < 1 or sample_size < 1:
            raise ValueError('dimensions, count, and sample_size must be positive')
        self.dimensions, self.expected_count = dimensions, int(expected_count)
        self.estimator = ('exact' if expected_count <= exact_limit else 'sampled') if estimator == 'auto' else estimator
        self.probability = min(1., sample_size / expected_count) if self.estimator == 'sampled' else 1.
        self.seed, self.quantiles = int(seed), quantiles
        self.count = self.sample_count = 0
        self.mean = np.zeros(dimensions, np.float64)
        self.m2 = np.zeros(dimensions, np.float64)
        self.low = np.full(dimensions, np.inf)
        self.high = np.full(dimensions, -np.inf)
        self._tmp = tempfile.TemporaryDirectory(prefix='b1k-stats-', dir=scratch)
        self._files = [open(Path(self._tmp.name) / f'{i}.f64', 'wb') for i in range(dimensions)] if quantiles else []

    def _selected(self, n):
        if self.probability == 1:
            return slice(None)
        x = np.arange(self.count, self.count + n, dtype=np.uint64) + np.uint64(self.seed)
        # SplitMix64: fixed across platforms, batching, and NumPy RNG versions.
        x += np.uint64(0x9e3779b97f4a7c15)
        x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
        x ^= x >> 31
        return (x >> 11).astype(np.float64) < self.probability * 2**53

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.dimensions)
        n = len(values)
        if not n:
            return
        if not np.isfinite(values).all():
            raise ValueError('non-finite normalization input')
        if self.count + n > self.expected_count:
            raise ValueError('more rows than expected (check padding and selection)')
        mean = values.mean(axis=0)
        delta = mean - self.mean
        total = self.count + n
        self.m2 += np.square(values - mean).sum(axis=0) + delta**2 * (self.count * n / total)
        self.mean += delta * (n / total)
        self.low = np.minimum(self.low, values.min(axis=0))
        self.high = np.maximum(self.high, values.max(axis=0))
        if self.quantiles:
            chosen = values[self._selected(n)]
            for dim, f in enumerate(self._files):
                np.ascontiguousarray(chosen[:, dim]).tofile(f)
            self.sample_count += len(chosen)
        self.count = total

    def finish(self):
        if self.count != self.expected_count:
            raise ValueError(f'expected {self.expected_count} valid rows, received {self.count}')
        result = {'mean': self.mean.copy(), 'std': np.sqrt(self.m2 / self.count),
                  'min': self.low.copy(), 'max': self.high.copy()}
        if self.quantiles:
            if self.sample_count < min(2,self.expected_count):
                raise ValueError('quantile sample is empty/too small; increase sample_size or use exact')
            quantiles = []
            for f in self._files:
                f.flush()
                values = np.fromfile(f.name, dtype=np.float64)
                quantiles.append(np.quantile(values, [0.01, 0.99], overwrite_input=True, method='linear'))
                del values  # release this column before allocating the next one
            result['q01'], result['q99'] = np.asarray(quantiles).T
        return result

    def metadata(self):
        return {'version': VERSION, 'estimator': self.estimator, 'count': self.count,
                'quantile_count': self.sample_count, 'sampling_probability': self.probability,
                'sampling_seed': self.seed, 'std_correction': 0, 'std_eps': EPS,
                'quantile_eps': EPS, 'padding': 'excluded', 'quantile_method': 'linear'}

    def close(self):
        for f in self._files:
            f.close()
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def shuffled_positions(positions, length, seed, epoch):
    """Cycle-walk a keyed Feistel permutation; O(batch), no O(frames) randperm.

    Every frame appears once per epoch. As with a finite-state RNG this is a
    pseudorandom shuffle, not uniform over all length! possible permutations.
    """
    values = np.asarray(positions, dtype=np.uint64)
    if length < 1 or np.any(values >= length):
        raise ValueError('invalid frame permutation input')
    bits = max(1, ((length - 1).bit_length() + 1) // 2)
    mask = np.uint64((1 << bits) - 1)
    keys = np.random.default_rng(np.random.SeedSequence([seed, epoch])).integers(0, 2**32, 6, dtype=np.uint64)
    result, active = values.copy(), np.ones(len(values), dtype=bool)
    while active.any():
        x = result[active]
        left, right = x >> bits, x & mask
        for key in keys:
            f = right + key
            f = (f ^ (f >> 16)) * np.uint64(0x45d9f3b)
            f = (f ^ (f >> 16)) * np.uint64(0x45d9f3b)
            f ^= f >> 16
            left, right = right, (left ^ f) & mask
        result[active] = (left << bits) | right
        active = result >= length
    return result.astype(np.int64)


class PiFrameBatchSampler:
    """Pi-style shuffled frame epochs, drop incomplete batches, DDP slices.

    Constant-memory frame permutation, no per-worker sampler state. Epoch seeding
    makes resume directly addressable. Matches Pi's frame population, shuffled
    epochs and dropped incomplete batch, not its PyTorch RNG byte stream.
    """
    def __init__(self, length, batch_size, start_step, max_steps, seed, *, rank=0, world_size=1,
                 tag=False, loader_batch_size=None):
        if length < batch_size or batch_size % world_size or not 0 <= rank < world_size:
            raise ValueError('need at least a global batch and a valid DDP batch/rank')
        self.length, self.global_batch_size = int(length), int(batch_size)
        self.batch_size, self.rank = batch_size // world_size, rank
        self.start_step, self.max_steps, self.seed = start_step, max_steps, seed
        self.tag, self.loader_batch_size = tag, loader_batch_size or self.batch_size
        self.steps_per_epoch = self.length // self.global_batch_size

    def __iter__(self):
        for step in range(self.start_step, self.max_steps):
            epoch, batch = divmod(step, self.steps_per_epoch)
            offset = batch * self.global_batch_size + self.rank * self.batch_size
            values = shuffled_positions(np.arange(offset, offset + self.batch_size), self.length, self.seed, epoch).tolist()
            if self.tag:
                values = [(v, step, self.rank * self.batch_size + i) for i, v in enumerate(values)]
            for i in range(0, self.batch_size, self.loader_batch_size):
                yield values[i:i + self.loader_batch_size]

    def __len__(self):
        return (self.max_steps - self.start_step) * ((self.batch_size + self.loader_batch_size - 1) // self.loader_batch_size)
