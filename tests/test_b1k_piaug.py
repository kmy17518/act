"""--image-augment pi05 (b1k_augment.py): openpi pi05_b1k's training augmentation against augmax's own output, its
draws, training batches only, the option on resume, and per-rank draws under DDP."""

import colorsys
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from b1k_augment import (adjust_contrast, augment_batch, augment_generator, hsv_to_rgb, pi05_augment, rgb_to_hsv,
                         sample_params)
from b1k_training import load_checkpoint, parser, train
from test_b1k import tiny_root  # noqa: F401  (fixture)
from test_b1k_side import add_episodes, free_port, tiny_common, train_fn

FIXTURE = Path(__file__).parent / 'fixtures' / 'pi05_augmax_reference.npz'


def fixture_params(reference):
    """The fixture's augmax draws as b1k_augment parameters."""
    draw = lambda name: torch.from_numpy(reference[f'draw.{name}'])
    color = lambda group: {'apply': draw(f'{group}.apply').bool(), 'brightness': draw(f'{group}.brightness'),
                           'contrast': draw(f'{group}.contrast'), 'hue': draw(f'{group}.hue')}
    height, width = reference['input.base_0_rgb'].shape[1:3]
    return {'theta': draw('theta'), 'cy': draw('cy'), 'cx': draw('cx'),
            'crop': (int(height * 0.95), int(width * 0.95)), 'color': color('color'),
            'wrist_color': color('wrist_color')}


def chw(array):
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).permute(0, 3, 1, 2)


@pytest.mark.parametrize('channels_last', [False, True])
def test_pi05_augment_reproduces_augmax(channels_last):
    """scripts/b1k/compare_pi05_augment.py --fixture: openpi 93f8a4e's preprocess_observation(train=True) on augmax
    0.4.1 (jax 0.11.1) for non-square 20x26 frames. Given augmax's draws, the torch version matches every key."""
    reference = np.load(FIXTURE)
    images = {key: chw(reference[f'input.{key}'] / np.float32(255)) for key in
              ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb', 'goal_0_rgb')}
    cameras = torch.stack([images['base_0_rgb'], images['left_wrist_0_rgb'], images['right_wrist_0_rgb']], 1)
    params = fixture_params(reference)
    assert params['color']['apply'].any() and not params['color']['apply'].all()  # both branches of the jitter
    assert params['wrist_color']['apply'].any() and not params['wrist_color']['apply'].all()
    out, goal = pi05_augment(cameras, images['goal_0_rgb'][:, None], [False, True, True], params, channels_last)
    for got, key in ((out[:, 0], 'base_0_rgb'), (out[:, 1], 'left_wrist_0_rgb'), (out[:, 2], 'right_wrist_0_rgb'),
                     (goal[:, 0], 'goal_0_rgb')):
        torch.testing.assert_close(got, chw(reference[f'output.{key}']), atol=2e-5, rtol=0, msg=key)
    if channels_last:  # prepare_images' layout: each camera's (batch, 3, H, W) slice a dense channels-last block
        assert out.permute(1, 0, 3, 4, 2).is_contiguous() and goal.permute(1, 0, 3, 4, 2).is_contiguous()
        assert out[:, 1].is_contiguous(memory_format=torch.channels_last)


def test_draws_have_openpis_ranges_and_depend_only_on_seed_step_and_rank():
    params = sample_params(4096, (224, 224), augment_generator(3, 7, 0))
    assert params['crop'] == (212, 212)
    for value, limit in ((params['theta'], math.radians(5)), (params['cy'], 6.0), (params['cx'], 6.0)):
        assert value.abs().max() <= limit and value.abs().max() > 0.98 * limit
        assert abs(float(value.std()) - limit / math.sqrt(3)) < 0.03 * limit
    for group in ('color', 'wrist_color'):
        for name, limit in (('brightness', 0.3), ('contrast', 0.4), ('hue', 0.1)):
            value = params[group][name]
            assert value.abs().max() <= limit and value.abs().max() > 0.98 * limit
        assert 0.45 < float(params[group]['apply'].float().mean()) < 0.55
    assert abs(float(np.corrcoef(params['color']['brightness'], params['wrist_color']['brightness'])[0, 1])) < 0.05
    flat = lambda p: torch.cat([p['theta'], p['cy'], p['color']['hue'], p['wrist_color']['hue']])
    same = sample_params(4096, (224, 224), augment_generator(3, 7, 0))
    assert torch.equal(flat(params), flat(same))
    for other in (augment_generator(4, 7, 0), augment_generator(3, 8, 0), augment_generator(3, 7, 1)):
        assert not torch.equal(flat(params), flat(sample_params(4096, (224, 224), other)))


def test_color_space_and_identity_parameters():
    generator = torch.Generator().manual_seed(0)
    pixels = torch.rand(2, 1, 3, 9, 11, generator=generator)
    pixels[0, 0, :, 0, :3] = torch.tensor([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.4, 0.4, 0.1]]).T  # gray, black, tie
    hue, saturation, value = rgb_to_hsv(pixels)
    for index in np.ndindex(2, 1, 9, 11):
        rgb = pixels[index[0], index[1], :, index[2], index[3]].tolist()
        want = colorsys.rgb_to_hsv(*rgb)
        got = (float(hue[index]) % 1.0, float(saturation[index]), float(value[index]))
        assert np.allclose(got, want, atol=1e-6), (rgb, got, want)
    torch.testing.assert_close(hsv_to_rgb(hue, saturation, value), pixels, atol=2e-6, rtol=0)
    ramp = torch.linspace(0, 1, 101)
    torch.testing.assert_close(adjust_contrast(ramp, torch.tensor(0.0)), ramp)
    for contrast in (-0.4, 0.4):  # monotone, fixes 0, 0.5 and 1
        curve = adjust_contrast(ramp, torch.tensor(contrast))
        assert (curve.diff() > 0).all() and torch.allclose(curve[[0, 50, 100]], torch.tensor([0.0, 0.5, 1.0]),
                                                            atol=1e-6)
    zero = torch.zeros(2)
    still = {'theta': zero, 'cy': zero, 'cx': zero, 'crop': (9, 11),
             'color': {'apply': torch.ones(2, dtype=torch.bool), 'brightness': zero, 'contrast': zero, 'hue': zero},
             'wrist_color': {'apply': torch.zeros(2, dtype=torch.bool), 'brightness': zero + 0.3, 'contrast': zero,
                             'hue': zero}}
    cameras = torch.rand(2, 3, 3, 9, 11, generator=generator)
    out, goal = pi05_augment(cameras, pixels, [False, True, True], still, channels_last=False)
    torch.testing.assert_close(out, cameras, atol=2e-6, rtol=0)
    torch.testing.assert_close(goal, pixels, atol=2e-6, rtol=0)


def augmented_calls(monkeypatch):
    import b1k_training
    calls = []

    def recorded(images, goal, wrist, seed, step, rank, channels_last):
        calls.append({'step': step, 'rank': rank, 'batch': images.shape[0], 'goal': goal is not None})
        return augment_batch(images, goal, wrist, seed, step, rank, channels_last)

    monkeypatch.setattr(b1k_training, 'augment_batch', recorded)
    return calls


IMAGE = ['--regime', 'image', '--goal-views', 'zed_link', '--goal-fusion', 'early', '--goal-stem-init', 'copy:0.5']
PIAUG = ['--image-augment', 'pi05']


def test_training_batches_only_and_the_option_on_resume(tiny_root, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    calls = augmented_calls(monkeypatch)
    add_episodes(tiny_root, 9, shift=0.5)
    common = tiny_common(tiny_root) + IMAGE + ['--task-names', 'first', '--val-fraction', '0.1', '--val-every', '2',
                                               '--val-batches', '2', '--val-batch-size', '3']
    run = tmp_path / 'run'
    checkpoint = load_checkpoint(train_fn(common + PIAUG + ['--max-steps', '2', '--output-dir', str(run)]))
    assert checkpoint['model_config']['image_augment'] == 'pi05'
    assert checkpoint['train_config']['image_augment'] == 'pi05'
    from b1k_server import PolicyPredictor
    PolicyPredictor(checkpoint, 'cpu')  # serving builds the policy from a model_config with the training-only key
    # validation ran (3-sample batches) without augmentation: one call per training step, with the 2-sample batch
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    assert any(key.startswith('val/') for record in records for key in record)
    assert calls == [{'step': step, 'rank': 0, 'batch': 2, 'goal': True} for step in (1, 2)]
    calls.clear()
    resumed = load_checkpoint(train_fn(common + ['--max-steps', '4', '--output-dir', str(run),
                                                 '--resume', str(run / 'step_00000002.pt')]))
    assert resumed['model_config']['image_augment'] == 'pi05' and [c['step'] for c in calls] == [3, 4]
    with pytest.raises(ValueError, match='--image-augment differs from checkpoint'):
        train_fn(common + ['--max-steps', '5', '--output-dir', str(tmp_path / 'conflict'), '--image-augment', 'none',
                           '--resume', str(run / 'step_00000004.pt')])

    plain = train_fn(common + ['--max-steps', '2', '--output-dir', str(tmp_path / 'plain')])
    assert load_checkpoint(plain)['model_config']['image_augment'] == 'none'
    old = torch.load(plain, weights_only=True)
    del old['model_config']['image_augment']
    legacy = tmp_path / 'legacy.pt'
    torch.save(old, legacy)
    calls.clear()
    again = load_checkpoint(train_fn(common + ['--max-steps', '3', '--output-dir', str(tmp_path / 'plain'),
                                               '--resume', str(legacy)]))
    assert 'image_augment' not in again['model_config'] and calls == []  # checkpoints before the option: none
    with pytest.raises(ValueError, match='--image-augment differs from checkpoint'):
        train_fn(common + PIAUG + ['--max-steps', '4', '--output-dir', str(tmp_path / 'conflict2'),
                                   '--resume', str(legacy)])


def test_resume_reproduces_the_augmented_run_exactly(tiny_root, tmp_path):
    torch.set_num_threads(1)
    common = tiny_common(tiny_root) + IMAGE + ['--save-every', '1', '--dropout', '0']
    full = load_checkpoint(train_fn(common + PIAUG + ['--max-steps', '4', '--output-dir', str(tmp_path / 'full')]))
    split = tmp_path / 'split'
    train_fn(common + PIAUG + ['--max-steps', '2', '--output-dir', str(split), '--loader-batch-size', '1'])
    resumed = load_checkpoint(train_fn(common + ['--max-steps', '4', '--output-dir', str(split),
                                                 '--resume', str(split / 'step_00000002.pt')]))
    for key in full['model']:
        torch.testing.assert_close(resumed['model'][key], full['model'][key], atol=0, rtol=0, msg=key)
    plain = load_checkpoint(train_fn(common + ['--max-steps', '4', '--output-dir', str(tmp_path / 'plain')]))
    assert any(not torch.equal(plain['model'][key], full['model'][key]) for key in full['model'])


def augment_rank_worker(rank, world_size, port, argv, record):
    """One gloo rank of `train`, recording each augment_batch call's step, rank and batch, and digests of its draws
    and of its input and output images."""
    import b1k_training
    os.environ.update(WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank), MASTER_ADDR='127.0.0.1',
                      MASTER_PORT=str(port))
    torch.set_num_threads(1)
    calls = []

    def recorded(images, goal, wrist, seed, step, rank_, channels_last):
        draws = torch.rand(images.shape[0], 11, generator=augment_generator(seed, step, rank_))
        out = augment_batch(images, goal, wrist, seed, step, rank_, channels_last)
        pixels = hashlib.sha256()
        for tensor in (images, goal, *out):
            pixels.update(tensor.contiguous().numpy().tobytes())
        calls.append({'step': step, 'rank': rank_, 'batch': images.shape[0],
                      'draws': hashlib.sha256(draws.numpy().tobytes()).hexdigest()[:16],
                      'pixels': pixels.hexdigest()[:16]})
        return out

    b1k_training.augment_batch = recorded
    returned = train(parser().parse_args(argv))
    Path(record.format(rank=rank)).write_text(json.dumps({'calls': calls, 'returned': str(returned)}))


def run_augment_ranks(tmp_path, argv, name, timeout=600):
    record = str(tmp_path / f'{name}-rank{{rank}}.json')
    context = mp.start_processes(augment_rank_worker, args=(2, free_port(), argv, record), nprocs=2,
                                 start_method='spawn', join=False)
    deadline = time.monotonic() + timeout
    while not context.join(timeout=5):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            raise TimeoutError(f'{name}: the two ranks did not finish within {timeout} s')
    return [json.loads(Path(record.format(rank=rank)).read_text()) for rank in range(2)]


def test_two_gloo_ranks_augment_their_own_halves_and_a_resume_reproduces_them(tiny_root, tmp_path):
    """Each rank draws its own augmentation for its half of the batch; a resume reproduces every rank's augmented
    images exactly. (Ranks > 0 reseed their torch RNG on resume, so DDP weights are not compared.)"""
    common = tiny_common(tiny_root) + IMAGE + PIAUG + ['--batch-size', '4', '--save-every', '2', '--dropout', '0']
    full = run_augment_ranks(tmp_path, common + ['--max-steps', '4', '--output-dir', str(tmp_path / 'full')], 'full')
    for rank, result in enumerate(full):
        assert [(c['step'], c['rank'], c['batch']) for c in result['calls']] == [(s, rank, 2) for s in range(1, 5)]
    assert not {c['draws'] for c in full[0]['calls']} & {c['draws'] for c in full[1]['calls']}
    split = tmp_path / 'split'
    run_augment_ranks(tmp_path, common + ['--max-steps', '2', '--output-dir', str(split)], 'first')
    resumed = run_augment_ranks(tmp_path, common + ['--max-steps', '4', '--output-dir', str(split), '--resume',
                                                    str(split / 'step_00000002.pt')], 'resumed')
    for rank in range(2):
        assert resumed[rank]['calls'] == full[rank]['calls'][2:]
    got = load_checkpoint(split / 'step_00000004.pt')
    assert got['model_config']['image_augment'] == 'pi05' and got['train_config']['world_size'] == 2
