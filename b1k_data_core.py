"""Portable B1K data primitives. Vendored identically in ACT, DP and openpi.

Version 2: linear exact quantiles (disk-backed), or reproducible Bernoulli
subsampling for quantiles only; all moments include every valid value.
No model imports, source writes, padding inference, or quantile widening.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import struct
from concurrent.futures import ThreadPoolExecutor

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
    """Exact moments and disk-backed quantiles; callers exclude padding first.

    zstd scratch contains the original float64 bits. Quantiles reconstruct one
    column per quantile worker and use NumPy's unchanged linear interpolation.
    float64 scratch remains selectable. A single optional writer overlaps lossless
    compression with the next update, retaining at most two encoded batches.
    """

    def __init__(
        self,
        dimensions,
        expected_count,
        *,
        estimator="auto",
        sample_size=10_000_000,
        exact_limit=500_000_000,
        seed=0,
        scratch=None,
        quantiles=True,
        storage=None,
        quantile_workers=None,
        async_write=None,
    ):
        if estimator not in ("auto", "exact", "sampled"):
            raise ValueError("estimator must be auto, exact, or sampled")
        if dimensions < 1 or expected_count < 1 or sample_size < 1:
            raise ValueError("dimensions, count, and sample_size must be positive")
        self.storage = (
            os.environ.get("B1K_STATS_STORAGE", "zstd") if storage is None else storage
        )
        self.quantile_workers = int(
            os.environ.get("B1K_STATS_QUANTILE_WORKERS", "1")
            if quantile_workers is None
            else quantile_workers
        )
        if self.storage not in ("float64", "zstd") or self.quantile_workers < 1:
            raise ValueError(
                "storage must be float64 or zstd; quantile_workers must be positive"
            )
        if async_write is None:
            flag = os.environ.get("B1K_STATS_ASYNC_WRITE", "0")
            if flag not in ("0", "1"):
                raise ValueError("B1K_STATS_ASYNC_WRITE must be 0 or 1")
            async_write = flag == "1"
        self.dimensions, self.expected_count = dimensions, int(expected_count)
        self.estimator = (
            ("exact" if expected_count <= exact_limit else "sampled")
            if estimator == "auto"
            else estimator
        )
        self.probability = (
            min(1.0, sample_size / expected_count)
            if self.estimator == "sampled"
            else 1.0
        )
        self.seed, self.quantiles = int(seed), quantiles
        self.count = self.sample_count = 0
        self.mean = np.zeros(dimensions, np.float64)
        self.m2 = np.zeros(dimensions, np.float64)
        self.low = np.full(dimensions, np.inf)
        self.high = np.full(dimensions, -np.inf)
        self._codec = None
        if quantiles and self.storage == "zstd":
            import pyarrow as pa

            self._codec = pa.Codec("zstd", compression_level=1)
        self._tmp = tempfile.TemporaryDirectory(prefix="b1k-stats-", dir=scratch)
        self._files = (
            [
                open(Path(self._tmp.name) / f"{i}.values", "wb")
                for i in range(dimensions)
            ]
            if quantiles
            else []
        )
        self._writer = (
            ThreadPoolExecutor(max_workers=1)
            if self._codec is not None and async_write
            else None
        )
        self._pending = None

    def _selected(self, n):
        if self.probability == 1:
            return slice(None)
        x = np.arange(self.count, self.count + n, dtype=np.uint64) + np.uint64(
            self.seed
        )
        # SplitMix64: fixed across platforms, batching, and NumPy RNG versions.
        x += np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> 27)) * np.uint64(0x94D049BB133111EB)
        x ^= x >> 31
        return (x >> 11).astype(np.float64) < self.probability * 2**53

    def _write_columns(self, encoded):
        for dim, f in enumerate(self._files):
            if self._codec is None:
                encoded[dim].tofile(f)
            else:
                payload = self._codec.compress(encoded[dim])
                f.write(struct.pack("<QQ", encoded.shape[1], len(payload)))
                f.write(payload)

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1, self.dimensions)
        n = len(values)
        if not n:
            return
        if not np.isfinite(values).all():
            raise ValueError("non-finite normalization input")
        if self.count + n > self.expected_count:
            raise ValueError("more rows than expected (check padding and selection)")
        mean = values.mean(axis=0)
        delta = mean - self.mean
        total = self.count + n
        centered = values - mean
        np.square(centered, out=centered)
        self.m2 += centered.sum(axis=0) + delta**2 * (self.count * n / total)
        self.mean += delta * (n / total)
        self.low = np.minimum(self.low, values.min(axis=0))
        self.high = np.maximum(self.high, values.max(axis=0))
        if self.quantiles:
            chosen = values[self._selected(n)]
            encoded = np.ascontiguousarray(chosen.T, dtype=np.float64)
            if self._writer is None:
                self._write_columns(encoded)
            else:
                if self._pending is not None:
                    self._pending.result()
                # One-dimensional inputs can already be contiguous after transpose.
                # Caller mutation after update must not change pending quantiles.
                if np.shares_memory(encoded, values):
                    encoded = encoded.copy()
                self._pending = self._writer.submit(self._write_columns, encoded)
            self.sample_count += len(chosen)
        self.count = total

    def _read_column(self, dim):
        path = self._files[dim].name
        if self._codec is None:
            values = np.fromfile(path, dtype=np.float64)
            if len(values) != self.sample_count:
                raise ValueError("truncated statistics scratch")
            return values
        values = np.empty(self.sample_count, dtype=np.float64)
        row = 0
        with open(path, "rb") as source:
            size_on_disk = os.fstat(source.fileno()).st_size
            while header := source.read(16):
                if len(header) != 16:
                    raise ValueError("truncated statistics scratch header")
                n, size = struct.unpack("<QQ", header)
                if n > self.sample_count - row or size > size_on_disk - source.tell():
                    raise ValueError("corrupt statistics scratch block")
                payload = source.read(size)
                raw = self._codec.decompress(payload, decompressed_size=n * 8)
                values[row : row + n] = np.frombuffer(raw, dtype=np.float64)
                row += n
        if row != self.sample_count:
            raise ValueError("truncated statistics scratch")
        return values

    def finish(self):
        if self._pending is not None:
            self._pending.result()
        if self.count != self.expected_count:
            raise ValueError(
                f"expected {self.expected_count} valid rows, received {self.count}"
            )
        result = {
            "mean": self.mean.copy(),
            "std": np.sqrt(self.m2 / self.count),
            "min": self.low.copy(),
            "max": self.high.copy(),
        }
        if self.quantiles:
            if self.sample_count < min(2, self.expected_count):
                raise ValueError(
                    "quantile sample is empty/too small; increase sample_size or use exact"
                )
            for f in self._files:
                f.flush()

            def column_quantiles(dim):
                return np.quantile(
                    self._read_column(dim),
                    [0.01, 0.99],
                    overwrite_input=True,
                    method="linear",
                )

            if self.quantile_workers == 1:
                quantiles = list(map(column_quantiles, range(self.dimensions)))
            else:
                with ThreadPoolExecutor(
                    max_workers=min(self.dimensions, self.quantile_workers)
                ) as pool:
                    quantiles = list(pool.map(column_quantiles, range(self.dimensions)))
            result["q01"], result["q99"] = np.asarray(quantiles).T
        return result

    def metadata(self):
        return {
            "version": VERSION,
            "estimator": self.estimator,
            "count": self.count,
            "quantile_count": self.sample_count,
            "sampling_probability": self.probability,
            "sampling_seed": self.seed,
            "std_correction": 0,
            "std_eps": EPS,
            "quantile_eps": EPS,
            "padding": "excluded",
            "quantile_method": "linear",
        }

    def close(self):
        try:
            if self._writer is not None:
                self._writer.shutdown(wait=True)
            if self._pending is not None:
                self._pending.result()
        finally:
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
