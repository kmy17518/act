"""Bit-level compatibility with the pre-compression estimator and safe scratch IO."""

import hashlib
import json
from pathlib import Path
import threading
import numpy as np
import pytest

try:
    from b1k_data_core import StreamingStats
except ImportError:
    try:
        from diffusion_policy.b1k.b1k_data_core import StreamingStats
    except ImportError:
        from openpi.shared.b1k_data_core import StreamingStats


def inputs(kind):
    rng = np.random.default_rng(71)
    x = rng.normal(size=(100003, 23)).astype(np.float32).astype(np.float64)
    if kind == "relative":
        x -= rng.normal(size=x.shape).astype(np.float32).astype(np.float64)
    if kind == "float64":
        x = rng.normal(size=x.shape)
    x[0, 0] = -0.0
    x[1, 0] = 0.0
    x[:, -1] = 1e-15 + (np.arange(len(x)) % 3) * 1e-21
    return x


def hashes(result):
    return {
        k: hashlib.sha256(np.asarray(v, dtype="<f8").tobytes()).hexdigest()
        for k, v in result.items()
    }


@pytest.mark.parametrize("kind", ["float32", "relative", "float64"])
@pytest.mark.parametrize("estimator", ["exact", "sampled"])
@pytest.mark.parametrize(
    "storage,asynchronous,workers",
    [
        ("float64", False, 1),
        ("float64", False, 2),
        ("zstd", False, 1),
        ("zstd", False, 2),
        ("zstd", True, 2),
    ],
)
def test_all_statistics_match_pre_compression_golden(
    kind, estimator, storage, asynchronous, workers
):
    golden = json.loads(
        (Path(__file__).with_name("b1k_stats_storage_golden.json")).read_text()
    )
    x = inputs(kind)
    with StreamingStats(
        23,
        len(x),
        estimator=estimator,
        sample_size=33001,
        seed=7,
        storage=storage,
        async_write=asynchronous,
        quantile_workers=workers,
    ) as s:
        for batch in np.array_split(x, 17):
            s.update(batch)
        assert hashes(s.finish()) == golden[kind + "/" + estimator]


def test_async_writer_copies_caller_memory_and_waits_before_finish():
    x = np.arange(100, dtype=np.float64).reshape(-1, 1)
    expected = np.quantile(x[:, 0], [0.01, 0.99])
    release = threading.Event()
    with StreamingStats(
        1, len(x), estimator="exact", storage="zstd", async_write=True
    ) as s:
        codec = s._codec

        class Delayed:
            def compress(self, values):
                assert release.wait(5)
                return codec.compress(values)

            def decompress(self, *args, **kwargs):
                return codec.decompress(*args, **kwargs)

        s._codec = Delayed()
        try:
            s.update(x)
            x[:] = 999
        finally:
            release.set()
        result = s.finish()
        assert result["q01"][0] == expected[0] and result["q99"][0] == expected[1]


def test_async_failure_propagates_and_cleans_scratch():
    class Broken:
        def compress(self, _):
            raise OSError("simulated disk/codec failure")

    s = StreamingStats(1, 10, estimator="exact", storage="zstd", async_write=True)
    temporary = Path(s._tmp.name)
    s._codec = Broken()
    with pytest.raises(OSError, match="simulated"):
        with s:
            s.update(np.arange(10))
            s.finish()
    assert not temporary.exists()


@pytest.mark.parametrize("storage", ["float64", "zstd"])
def test_truncated_scratch_never_produces_statistics(storage):
    with StreamingStats(
        1, 100, estimator="exact", storage=storage, async_write=False
    ) as s:
        s.update(np.arange(100))
        f = s._files[0]
        f.flush()
        f.truncate(f.tell() - 1)
        with pytest.raises(ValueError, match="scratch"):
            s.finish()


def test_mean_only_does_not_create_quantile_files():
    with StreamingStats(2, 3, quantiles=False, storage="zstd", async_write=True) as s:
        s.update([[1, 2], [3, 4], [5, 6]])
        r = s.finish()
        assert s._files == [] and s._writer is None
        np.testing.assert_array_equal(r["mean"], [3, 4])
    with pytest.raises(ValueError, match="positive"):
        StreamingStats(1, 1, quantile_workers=0)


def test_sampled_empty_blocks_preserve_exact_selected_values():
    outputs = []
    for storage in ("float64", "zstd"):
        with StreamingStats(
            1, 1000, estimator="sampled", sample_size=20, storage=storage
        ) as s:
            for value in range(1000):
                s.update([value])
            outputs.append(hashes(s.finish()))
    assert outputs[0] == outputs[1]
