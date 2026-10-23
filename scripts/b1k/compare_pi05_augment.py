#!/usr/bin/env python3
"""Compare `--image-augment pi05` (b1k_augment.py) with what it replicates: openpi 93f8a4e
models/model.py:preprocess_observation(train=True) on augmax 0.4.1.

Runs on CPU in a venv with jax, augmax==0.4.1, einops, numpy, scipy, torch and av (not the training venv):

  python scripts/b1k/compare_pi05_augment.py --video A.mp4 [--video B.mp4 ...] [--json OUT]

1. pixels: openpi's loop over image keys on a batch (base_0_rgb and the goal slot goal_0_rgb run augmax's
   crop/resize/rotate + ColorJitter chain, left/right_wrist_0_rgb its ColorJitter-only chain), augmax's draws for each
   sample re-derived from the same keys, and b1k_augment.pi05_augment with those draws: per key, the largest absolute
   pixel difference. The batch has video frames plus synthetic stress images (uint8 noise, so channel ties; gray;
   black; saturated colors);
2. draws: b1k_augment's draws over training steps against augmax's over as many keys (two-sample Kolmogorov-Smirnov
   test per parameter, jitter rates, head/wrist jitter correlation);
3. images: statistics of independently augmented frames both ways (mean absolute change, zero-fill fraction).

--fixture PATH writes the small augmax reference tests/test_b1k_piaug.py checks against (non-square frames, augmax's
draws and outputs).
"""
import argparse
import json
import math
from pathlib import Path
import sys

import augmax
import av
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import b1k_augment  # noqa: E402

KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb', 'goal_0_rgb')  # openpi IMAGE_KEYS + goal slot
SIZE = (224, 224)


def openpi_augment(rng, images):
    """openpi 93f8a4e preprocess_observation(train=True)'s loop; images {key: (B, H, W, 3)} in [0, 1]."""
    out = {}
    for key, image in images.items():
        image = (image * 2.0 - 1.0) / 2.0 + 0.5  # openpi's model inputs are in [-1, 1]
        transforms = []
        if 'wrist' not in key:
            height, width = image.shape[1:3]
            transforms += [augmax.RandomCrop(int(width * 0.95), int(height * 0.95)), augmax.Resize(width, height),
                           augmax.Rotate((-5, 5))]
        transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
        sub_rngs = jax.random.split(rng, image.shape[0])
        image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)
        out[key] = (image * 2.0 - 1.0) / 2.0 + 0.5
    return out


def jitter_draws(key):
    """augmax ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5).pixelwise's draws from its key: 5 keys for
    brightness, contrast, hue, saturation (result discarded) and one unused; the apply flag from the key itself."""
    keys = jax.random.split(key, 5)
    return {'apply': jax.random.bernoulli(key, 0.5),
            'brightness': jax.random.uniform(keys[0], minval=-0.3, maxval=0.3),
            'contrast': jax.random.uniform(keys[1], minval=-0.4, maxval=0.4),
            'hue': jax.random.uniform(keys[2], minval=-0.1, maxval=0.1)}


def crop_of(size):
    return int(size[0] * 0.95), int(size[1] * 0.95)


def augmax_draws(sub_rng, size=SIZE):
    """The draws augmax makes from one sample's key in openpi's two chains."""
    k_geometric, k_color = jax.random.split(sub_rng, 2)  # Chain -> [GeometricChain, ColorspaceChain]
    k_crop, _, k_rotate = jax.random.split(k_geometric, 3)  # one key per crop, resize, rotate
    crop = crop_of(size)
    limit = jnp.array([(size[0] - crop[0]) / 2, (size[1] - crop[1]) / 2])
    cy, cx = jax.random.uniform(k_crop, [2], minval=-limit, maxval=limit)
    theta = jax.random.bernoulli(k_rotate, 1.0) * jax.random.uniform(k_rotate, minval=math.radians(-5),
                                                                     maxval=math.radians(5))
    (k_wrist,) = jax.random.split(sub_rng, 1)  # wrist Chain -> [ColorspaceChain]
    return {'theta': theta, 'cy': cy, 'cx': cx, 'color': jitter_draws(jax.random.split(k_color, 1)[0]),
            'wrist_color': jitter_draws(jax.random.split(k_wrist, 1)[0])}


def torch_params(draws, size=SIZE):
    tensor = lambda value: torch.from_numpy(np.array(value))
    color = lambda d: {'apply': tensor(d['apply']).bool(), 'brightness': tensor(d['brightness']),
                       'contrast': tensor(d['contrast']), 'hue': tensor(d['hue'])}
    return {'theta': tensor(draws['theta']), 'cy': tensor(draws['cy']), 'cx': tensor(draws['cx']), 'crop': crop_of(size),
            'color': color(draws['color']), 'wrist_color': color(draws['wrist_color'])}


def load_frames(videos, per_video):
    """Frames spread over each video, resized as ACT's preprocess_image and quantized as its frame cache."""
    frames = []
    for video in videos:
        with av.open(str(video)) as container:
            decoded = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
        for index in np.linspace(0, len(decoded) - 1, per_video).round().astype(int):
            tensor = torch.from_numpy(decoded[index]).permute(2, 0, 1).float()[None] / 255.0
            tensor = F.interpolate(tensor, size=SIZE, mode='bilinear', align_corners=False, antialias=True)[0]
            frames.append((tensor.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy())
    return np.stack(frames)


def stress_frames(rng):
    noise = rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)  # many channel ties: argmax order matters
    coarse = rng.integers(0, 4, (*SIZE, 3), dtype=np.uint8) * 85  # mostly ties, 0 and 255 included
    gray = np.repeat(np.linspace(0, 255, SIZE[1]).round().astype(np.uint8)[None, :, None], SIZE[0], 0).repeat(3, 2)
    black = np.zeros((*SIZE, 3), np.uint8)
    blocks = np.zeros((*SIZE, 3), np.uint8)
    colors = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (255, 0, 255), (255, 255, 255),
              (128, 128, 0), (3, 2, 1), (250, 250, 251)]
    for i, color in enumerate(colors):
        blocks[:, i * SIZE[1] // len(colors):(i + 1) * SIZE[1] // len(colors)] = color
    return np.stack([noise, coarse, gray, black, blocks])


def ours_augment(frames, params):
    """frames {key: (B, H, W, 3)} -> b1k_augment.pi05_augment with ACT's layout (cameras zed_link, left, right)."""
    to_chw = lambda key: torch.from_numpy(np.asarray(frames[key])).permute(0, 3, 1, 2)
    images = torch.stack([to_chw('base_0_rgb'), to_chw('left_wrist_0_rgb'), to_chw('right_wrist_0_rgb')], 1)
    out, goal = b1k_augment.pi05_augment(images, to_chw('goal_0_rgb')[:, None], (False, True, True), params)
    return {'base_0_rgb': out[:, 0], 'left_wrist_0_rgb': out[:, 1], 'right_wrist_0_rgb': out[:, 2],
            'goal_0_rgb': goal[:, 0]}


def batch_of(frames, shift=1):
    """One image per key, each key a different frame of the pool."""
    count = len(frames)
    return {key: (frames[(np.arange(count) + i * shift) % count].astype(np.float32) / 255.0)
            for i, key in enumerate(KEYS)}


def compare_pixels(pool, seed):
    images = batch_of(pool)
    rng = jax.random.PRNGKey(seed)
    reference = {key: torch.from_numpy(np.array(value)).permute(0, 3, 1, 2)
                 for key, value in openpi_augment(rng, {k: jnp.asarray(v) for k, v in images.items()}).items()}
    draws = jax.vmap(augmax_draws)(jax.random.split(rng, len(pool)))
    ours = ours_augment(images, torch_params(draws))
    result = {'batch': len(pool), 'jitter_applied': int(np.array(draws['color']['apply']).sum()),
              'wrist_jitter_applied': int(np.array(draws['wrist_color']['apply']).sum())}
    for key in KEYS:
        source = torch.from_numpy(images[key]).permute(0, 3, 1, 2)
        diff = (ours[key] - reference[key]).abs()
        result[key] = {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
                       'over_half_level': float((diff > 0.5 / 255).float().mean()),
                       'augmax_mean_change': float((reference[key] - source).abs().mean())}
    return result


def compare_draws(samples, seed, batch=256):
    reference = jax.tree_util.tree_map(np.array, jax.vmap(augmax_draws)(
        jax.random.split(jax.random.PRNGKey(seed + 1), samples)))
    params = [b1k_augment.sample_params(batch, SIZE, b1k_augment.augment_generator(seed, step, 0), 'cpu')
              for step in range(math.ceil(samples / batch))]
    ours = lambda get: np.concatenate([get(p).numpy() for p in params])[:samples].astype(np.float64)
    table = {}
    for name, get, scale in [('theta_degrees', lambda p: p['theta'], 180 / math.pi),
                             ('crop_offset_y', lambda p: p['cy'], 1), ('crop_offset_x', lambda p: p['cx'], 1)] + [
            (f'{prefix}{field}', lambda p, g=group, f=field: p[g][f], 1)
            for group, prefix in (('color', ''), ('wrist_color', 'wrist_')) for field in ('brightness', 'contrast',
                                                                                          'hue', 'apply')]:
        a, b = get(reference).astype(np.float64) * scale, ours(get) * scale
        row = {'augmax': [float(a.min()), float(a.mean()), float(a.std()), float(a.max())],
               'ours': [float(b.min()), float(b.mean()), float(b.std()), float(b.max())]}
        if not name.endswith('apply'):
            row['ks_pvalue'] = float(stats.ks_2samp(a, b).pvalue)
        table[name] = row
    table['head_wrist_apply_correlation'] = {
        'augmax': float(np.corrcoef(reference['color']['apply'], reference['wrist_color']['apply'])[0, 1]),
        'ours': float(np.corrcoef(ours(lambda p: p['color']['apply']), ours(lambda p: p['wrist_color']['apply']))[0, 1])}
    return {'samples': samples, 'draws': table}


def compare_images(pool, seed, rounds):
    """Independent draws both ways on the same frames: mean absolute change and zero-fill fraction per key type."""
    totals = {side: {kind: [] for kind in ('head_change', 'goal_change', 'wrist_change', 'head_zero_fill')}
              for side in ('augmax', 'ours')}
    for r in range(rounds):
        images = batch_of(pool, shift=r + 1)
        rng = jax.random.PRNGKey(seed + 100 + r)
        reference = {key: torch.from_numpy(np.array(value)).permute(0, 3, 1, 2)
                     for key, value in openpi_augment(rng, {k: jnp.asarray(v) for k, v in images.items()}).items()}
        generator = b1k_augment.augment_generator(seed, 10_000 + r, 0)
        ours = ours_augment(images, b1k_augment.sample_params(len(pool), SIZE, generator, 'cpu'))
        for side, out in (('augmax', reference), ('ours', ours)):
            source = {key: torch.from_numpy(images[key]).permute(0, 3, 1, 2) for key in KEYS}
            change = lambda key: (out[key] - source[key]).abs().mean(dim=(1, 2, 3))
            totals[side]['head_change'].append(change('base_0_rgb'))
            totals[side]['goal_change'].append(change('goal_0_rgb'))
            totals[side]['wrist_change'].append(torch.cat([change('left_wrist_0_rgb'), change('right_wrist_0_rgb')]))
            filled = (out['base_0_rgb'].amax(1) == 0) & (source['base_0_rgb'].amax(1) > 0)
            totals[side]['head_zero_fill'].append(filled.float().mean(dim=(1, 2)))
    summary = {}
    for side, kinds in totals.items():
        summary[side] = {}
        for kind, values in kinds.items():
            values = torch.cat(values).double()
            summary[side][kind] = {'mean': float(values.mean()),
                                   'stderr': float(values.std() / math.sqrt(len(values)))}
    return {'samples_per_key': rounds * len(pool), **summary}


def write_fixture(path, seed, size=(20, 26), batch=6):
    """Noise and gradient frames of a non-square size, augmax's draws for them and openpi's augmented images."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 255, size[1])[None, :, None] * np.linspace(0.2, 1, size[0])[:, None, None] * np.ones(3)
    frames = {key: rng.integers(0, 256, (batch, *size, 3), dtype=np.uint8) for key in KEYS}
    for i, key in enumerate(KEYS):
        frames[key][::2] = np.roll(ramp, 7 * i, axis=1).round().astype(np.uint8)
        frames[key][::2, ..., i % 3] = 255 - frames[key][::2, ..., i % 3]
    images = {key: value.astype(np.float32) / 255.0 for key, value in frames.items()}
    key = jax.random.PRNGKey(seed)
    reference = openpi_augment(key, {k: jnp.asarray(v) for k, v in images.items()})
    draws = jax.vmap(lambda k: augmax_draws(k, size))(jax.random.split(key, batch))
    flat = {'theta': draws['theta'], 'cy': draws['cy'], 'cx': draws['cx']}
    for group in ('color', 'wrist_color'):
        flat.update({f'{group}.{name}': value for name, value in draws[group].items()})
    arrays = {f'input.{k}': v for k, v in frames.items()}
    arrays.update({f'output.{k}': np.array(v) for k, v in reference.items()})
    arrays.update({f'draw.{k}': np.array(v) for k, v in flat.items()})
    np.savez_compressed(path, **arrays)
    ours = ours_augment(images, torch_params(draws, size))
    return max(float((ours[k].permute(0, 2, 3, 1) - torch.from_numpy(np.array(reference[k]))).abs().max()) for k in KEYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--video', action='append', type=Path, default=[])
    parser.add_argument('--fixture', type=Path, help='write the test fixture here and exit')
    parser.add_argument('--per-video', type=int, default=9)
    parser.add_argument('--samples', type=int, default=20_000)
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.fixture:
        print(f'{args.fixture}: largest difference {write_fixture(args.fixture, args.seed):.3g}')
        return
    if not args.video:
        parser.error('--video is required')
    frames = load_frames(args.video, args.per_video)
    pool = np.concatenate([frames, stress_frames(np.random.default_rng(args.seed))])
    report = {'jax': jax.__version__, 'augmax': augmax.__version__ if hasattr(augmax, '__version__') else
              __import__('importlib.metadata').metadata.version('augmax'), 'torch': torch.__version__,
              'frames': {'video': len(frames), 'stress': len(pool) - len(frames)}}
    report['pixels'] = compare_pixels(pool, args.seed)
    print(json.dumps({'pixels': report['pixels']}, indent=1), flush=True)
    report['draws'] = compare_draws(args.samples, args.seed)
    report['images'] = compare_images(frames, args.seed, args.rounds)
    print(json.dumps({k: report[k] for k in ('jax', 'augmax', 'torch', 'frames', 'draws', 'images')}, indent=1))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
