"""isg-train224: pi05's LR schedule, prepared (precomputed) startup, native-size frame caches."""
import json

import numpy as np
import pytest
import torch

from test_b1k import tiny_root  # noqa: F401  (fixture)
from b1k_data_core import pi_lr_factor
from b1k_dataset import B1KDataset
from b1k_cache_v2 import build_group
from b1k_frame_cache import selected_videos


def _common(root, cache, stats):
    return ['--dataset-path', str(root), '--batch-size', '2', '--num-workers', '0', '--device', 'cpu',
            '--chunk-size', '4', '--image-size', '16', '16', '--hidden-dim', '32', '--dim-feedforward', '64',
            '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4', '--no-pretrained-backbone', '--save-every', '2',
            '--frame-cache', str(cache), '--stats-file', str(stats), '--goal-views', 'zed_link', '--goal-fusion', 'late',
            '--action-representation', 'relative', '--normalization', 'quantile']


def _cache(root, tmp_path):
    native = B1KDataset(root, chunk_size=4, image_size=(16, 16))
    cache = tmp_path / 'cache'
    for path in selected_videos(native):
        build_group([path], cache, root, (16, 16))
    native.close()
    return cache


def test_pi_lr_factor_shape():
    assert pi_lr_factor(0, 1000, 100000) == pytest.approx(1 / 1001)
    assert pi_lr_factor(1000, 1000, 100000) == pytest.approx(1.0)
    assert pi_lr_factor(100000, 1000, 100000) == pytest.approx(0.1)
    assert pi_lr_factor(250000, 1000, 100000) == pytest.approx(0.1)
    mid = pi_lr_factor(1000 + 49500, 1000, 100000)
    assert mid == pytest.approx(0.55)


def test_prepare_only_then_require_prepared_with_pi_cosine_resume(tiny_root, tmp_path):
    from b1k_training import train, parser, load_checkpoint
    cache = _cache(tiny_root, tmp_path)
    stats = tmp_path / 'prepared' / 'stats.json'
    common = _common(tiny_root, cache, stats)
    schedule = ['--lr-schedule', 'pi_cosine', '--lr-warmup-steps', '2', '--lr-decay-steps', '6']
    # Without the prepared statistics a --require-prepared run refuses to start.
    with pytest.raises(FileNotFoundError, match='prepare-only'):
        train(parser().parse_args(common + schedule + ['--output-dir', str(tmp_path / 'early'), '--max-steps', '2',
                                                       '--require-prepared']))
    assert train(parser().parse_args(common + schedule + ['--output-dir', str(tmp_path / 'prep'), '--max-steps', '4',
                                                          '--prepare-only'])) is None
    assert stats.is_file() and not (tmp_path / 'prep' / 'run.json').exists()
    record = json.loads(stats.read_text())
    assert record['gripper_action'] == 'absolute' and record['estimator'] == 'exact'
    tables = list((cache.parent / (cache.name + '-tables')).glob('*/complete.json'))
    assert tables and list((cache.parent / (cache.name + '-tables')).glob('*/worker-selection/*/complete.json'))
    full, split = tmp_path / 'full', tmp_path / 'split'
    train(parser().parse_args(common + schedule + ['--output-dir', str(full), '--max-steps', '4', '--require-prepared']))
    rows = [json.loads(line) for line in (full / 'metrics.jsonl').read_text().splitlines()]
    peak = 1e-5
    assert [r['lr'] for r in rows] == pytest.approx([peak * pi_lr_factor(i, 2, 6) for i in range(4)])
    saved = load_checkpoint(full / 'step_00000004.pt')
    assert saved['model_config']['lr_schedule'] == {'kind': 'pi_cosine', 'warmup_steps': 2, 'decay_steps': 6,
                                                    'end_ratio': 0.1}
    # Resuming continues the schedule exactly (the schedule is a function of the step, restored from the checkpoint).
    train(parser().parse_args(common + schedule + ['--output-dir', str(split), '--max-steps', '2', '--require-prepared']))
    path = train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '4', '--require-prepared',
                                               '--resume', str(split / 'step_00000002.pt')]))
    resumed = load_checkpoint(path)
    for key in saved['model']:
        torch.testing.assert_close(saved['model'][key], resumed['model'][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='lr-decay-steps'):
        train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '5', '--lr-decay-steps', '9',
                                            '--resume', str(split / 'step_00000004.pt')]))


def test_constant_schedule_flags_need_pi_cosine(tiny_root, tmp_path):
    from b1k_training import train, parser
    cache = _cache(tiny_root, tmp_path)
    with pytest.raises(ValueError, match='needs --lr-schedule pi_cosine'):
        train(parser().parse_args(_common(tiny_root, cache, tmp_path / 's.json') +
                                  ['--output-dir', str(tmp_path / 'o'), '--max-steps', '2', '--lr-warmup-steps', '5',
                                   '--prepare-only']))


def test_native_size_frames_skip_the_resize(tiny_root, tmp_path, monkeypatch):
    """A v1 cache at the videos' own size stores the decoded pixels unchanged (no float round trip), byte-identical
    to the resize path's output."""
    import b1k_frame_cache
    from b1k_frame_cache import build_entry, cache_stem, FrameCacheReader
    native = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16))
    calls = []
    original = b1k_frame_cache.preprocess_image
    monkeypatch.setattr(b1k_frame_cache, 'preprocess_image', lambda *a, **k: calls.append(1) or original(*a, **k))
    cache = tmp_path / 'v1'
    try:
        for path in selected_videos(native):
            build_entry(path, cache_stem(cache, native.root, path), (16, 16), native.root)
        assert not calls
        reader = FrameCacheReader(cache, native.root, (16, 16))
        for path in selected_videos(native):
            import av
            with av.open(str(path)) as container:
                for i, frame in enumerate(container.decode(video=0)):
                    pixels = frame.to_ndarray(format='rgb24')
                    np.testing.assert_array_equal(reader.read(path, i), pixels)
                    np.testing.assert_array_equal(
                        pixels, b1k_frame_cache.quantize_image(original(pixels, (16, 16))))
        reader.close()
    finally:
        native.close()
