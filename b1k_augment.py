"""openpi pi05_b1k's training image augmentation, in torch on the GPU (`--image-augment pi05`).

Replicates openpi 93f8a4e `models/model.py:preprocess_observation(train=True)` with augmax 0.4.1 (the version of its
.venv-jax) on images in [0, 1] (224x224 in the B1K runs). openpi splits one key per sample and uses that same key
for every image of the sample, so all images that run the same augmax chain get the same draws:

- non-wrist images (the head camera zed_link and every goal image; openpi keys without "wrist"):
  `augmax.Chain(RandomCrop(212, 212), Resize(224, 224), Rotate((-5, 5)), ColorJitter(...))`, which augmax splits into
  `GeometricChain(crop, resize, rotate)` -- fused into one bilinear resampling with zero fill: output pixel (y, x),
  centered at (H/2 - 0.5, W/2 - 0.5), samples the input at s * R(theta) (y, x) + (cy, cx), s = 212 / 224,
  theta ~ U(-5, 5) degrees, cy, cx ~ U(-6, 6) -- then `ColorspaceChain(ColorJitter)`;
- wrist images (left/right realsense): `Chain(ColorJitter)` only. Its key is split differently, so the wrist color draws
  differ from the non-wrist ones; both wrists share theirs.

ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) keeps augmax's defaults hue=0.1 and p=0.5: with probability
0.5 per sample and chain, in HSV, the value gets brightness b ~ U(-0.3, 0.3) (v * (1 + b) for b < 0, else v * (1 - b) + b), then
contrast c ~ U(-0.4, 0.4) (augmax's invertible piecewise curve with slant tan((c + 1) * pi / 4)), the hue shifts by
U(-0.1, 0.1); the saturation draw changes nothing (augmax 0.4.1 discards the result of that adjustment).

The draws come from a CPU generator seeded by (seed, step, rank): a resume reproduces them on any GPU, and each DDP
rank draws its own.
"""
import hashlib
import logging
import math

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
IMAGE_AUGMENTS = ('none', 'pi05')
CROP_FRACTION = 0.95
ANGLE_DEGREES = 5.0
BRIGHTNESS, CONTRAST, HUE = 0.3, 0.4, 0.1
JITTER_PROBABILITY = 0.5
DRAWS = ('theta', 'cy', 'cx', 'apply', 'brightness', 'contrast', 'hue', 'wrist_apply', 'wrist_brightness',
         'wrist_contrast', 'wrist_hue')


def augment_generator(seed, step, rank):
    """The CPU generator of one training step's draws on one rank: the same numbers on every device and GPU model."""
    key = hashlib.sha256(f'pi05-augment/{int(seed)}/{int(step)}/{int(rank)}'.encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(key[:8], 'little') & ((1 << 63) - 1))


def sample_params(batch, size, generator, device='cpu'):
    """Each sample's draws as openpi makes them (geometry and color for the non-wrist images, color for the wrists),
    from one (batch, len(DRAWS)) uniform matrix drawn on the CPU and copied to `device` without a stream sync."""
    uniforms = torch.rand(batch, len(DRAWS), generator=generator)
    if torch.device(device).type == 'cuda':
        uniforms = uniforms.pin_memory().to(device, non_blocking=True)
    draws = dict(zip(DRAWS, uniforms.to(device).unbind(1)))
    height, width = size
    crop_h, crop_w = int(height * CROP_FRACTION), int(width * CROP_FRACTION)

    def span(name, limit):
        return (2 * draws[name] - 1) * limit

    def color(prefix):
        return {'apply': draws[prefix + 'apply'] < JITTER_PROBABILITY,
                'brightness': span(prefix + 'brightness', BRIGHTNESS), 'contrast': span(prefix + 'contrast', CONTRAST),
                'hue': span(prefix + 'hue', HUE)}

    return {'theta': span('theta', math.radians(ANGLE_DEGREES)), 'cy': span('cy', (height - crop_h) / 2),
            'cx': span('cx', (width - crop_w) / 2), 'crop': (crop_h, crop_w), 'color': color(''),
            'wrist_color': color('wrist_')}


def sampling_grid(params, size, device):
    """grid_sample grid (batch, H, W, 2) of augmax's fused RandomCrop -> Resize -> Rotate for each sample."""
    height, width = size
    crop_h, crop_w = params['crop']
    sy, sx = crop_h / height, crop_w / width
    y = torch.arange(height, device=device, dtype=torch.float32) - (height / 2 - 0.5)
    x = torch.arange(width, device=device, dtype=torch.float32) - (width / 2 - 0.5)
    y, x = torch.meshgrid(y, x, indexing='ij')
    cos = torch.cos(params['theta']).float().view(-1, 1, 1)
    sin = torch.sin(params['theta']).float().view(-1, 1, 1)
    y_in = sy * (cos * y + sin * x) + params['cy'].float().view(-1, 1, 1) + (height / 2 - 0.5)
    x_in = sx * (-sin * y + cos * x) + params['cx'].float().view(-1, 1, 1) + (width / 2 - 0.5)
    return torch.stack((2 * x_in / (width - 1) - 1, 2 * y_in / (height - 1) - 1), dim=-1)


def warp(images, grid):
    """images (batch, k, 3, H, W) resampled bilinearly with zero fill on each sample's grid."""
    batch, count = images.shape[:2]
    flat = images.reshape(batch * count, *images.shape[2:])
    flat_grid = grid.unsqueeze(1).expand(batch, count, *grid.shape[1:]).reshape(batch * count, *grid.shape[1:])
    out = F.grid_sample(flat, flat_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return out.reshape(images.shape)


def rgb_to_hsv(rgb):
    """augmax.utils.rgb_to_hsv per pixel: rgb (..., 3, H, W) -> hue, saturation, value (..., H, W)."""
    value = rgb.max(dim=-3).values
    argmax = torch.argmax(rgb, dim=-3)  # the first maximum, as jnp.argmax
    spread = value - rgb.min(dim=-3).values
    second = torch.gather(rgb, -3, ((argmax + 1) % 3).unsqueeze(-3)).squeeze(-3)
    third = torch.gather(rgb, -3, ((argmax + 2) % 3).unsqueeze(-3)).squeeze(-3)
    safe = torch.where(spread == 0, torch.ones_like(spread), spread)
    hue = torch.where(spread == 0, torch.zeros_like(spread), (2 * argmax + (second - third) / safe) / 6)
    saturation = torch.where(value == 0, torch.zeros_like(value), spread / torch.where(value == 0, torch.ones_like(value), value))
    return hue, saturation, value


def hsv_to_rgb(hue, saturation, value):
    """augmax.utils.hsv_to_rgb: n = (5, 3, 1), k = (n + 6 h) mod 6, channel = v - v s clip(min(k, 4 - k), 0, 1)."""
    n = torch.arange(5.0, 0.0, -2.0, device=hue.device).view(*([1] * (hue.dim() - 2)), 3, 1, 1)  # no host copy
    k = torch.remainder(n + hue.unsqueeze(-3) * 6, 6)
    return value.unsqueeze(-3) - value.unsqueeze(-3) * saturation.unsqueeze(-3) * torch.clamp(
        torch.minimum(k, 4 - k), 0, 1)


def adjust_brightness(value, brightness):
    """augmax.functional.colorspace.adjust_brightness (forward)."""
    return torch.where(brightness < 0, value * (1 + brightness), value * (1 - brightness) + brightness)


def adjust_contrast(value, contrast):
    """augmax.functional.colorspace.adjust_contrast (forward): the piecewise curve with slant tan((c + 1) pi / 4)."""
    slant = torch.tan((contrast + 1) * (math.pi / 4))
    p1 = (slant - slant ** 2) / (2 * (1 - slant ** 2))
    p2 = 1 - p1
    middle = slant * (value - 0.5) + 0.5
    return torch.where(value < p1, value / slant, torch.where(value > p2, value / slant + 1 - 1 / slant, middle))


def color_jitter(images, color):
    """augmax ColorJitter on images (batch, k, 3, H, W) with one draw per sample shared by its k images."""
    hue, saturation, value = rgb_to_hsv(images)
    view = (-1,) + (1,) * (hue.dim() - 1)  # one draw per sample, broadcast over its images and pixels
    value = adjust_brightness(value, color['brightness'].float().view(view))
    value = adjust_contrast(value, color['contrast'].float().view(view))
    hue = hue + color['hue'].float().view(view)
    jittered = hsv_to_rgb(hue, saturation, value)
    return torch.where(color['apply'].view(view + (1,)), jittered, images)


def pi05_augment(images, goal, wrist, params):
    """The augmented (images, goal) in the inputs' memory layouts: images (batch, cameras, 3, H, W) in [0, 1] with
    `wrist[c]` per camera, goal (batch, views, 3, H, W) or None (always non-wrist, like openpi's goal keys)."""
    grid = sampling_grid(params, tuple(images.shape[-2:]), images.device)
    head = [index for index, is_wrist in enumerate(wrist) if not is_wrist]
    hands = [index for index, is_wrist in enumerate(wrist) if is_wrist]
    # slices only: indexing with a list of cameras would copy an index tensor to the GPU and synchronize
    nonwrist = torch.cat([images[:, i:i + 1] for i in head] + ([goal] if goal is not None else []), dim=1)
    nonwrist = color_jitter(warp(nonwrist.float(), grid), params['color'])
    out = torch.empty_like(images)
    for k, i in enumerate(head):
        out[:, i:i + 1] = nonwrist[:, k:k + 1]
    if hands:
        jittered = color_jitter(torch.cat([images[:, i:i + 1] for i in hands], dim=1).float(), params['wrist_color'])
        for k, i in enumerate(hands):
            out[:, i:i + 1] = jittered[:, k:k + 1]
    if goal is None:
        return out, None
    new_goal = torch.empty_like(goal)
    new_goal.copy_(nonwrist[:, len(head):])
    return out, new_goal


_COMPILED = {}


def compiled_pi05_augment():
    """pi05_augment through torch.compile (inductor, static shapes): its elementwise passes fused into a few kernels."""
    if 'augment' not in _COMPILED:
        _COMPILED['augment'] = torch.compile(pi05_augment, dynamic=False, fullgraph=True)
    return _COMPILED['augment']


def augment_batch(images, goal, wrist, seed, step, rank, compiled=False):
    """`--image-augment pi05` for one training batch: draws from (seed, step, rank), then pi05_augment, compiled when
    the run compiles its policy on CUDA (eager if compiling fails: the same augmentation, only slower)."""
    params = sample_params(images.shape[0], tuple(images.shape[-2:]), augment_generator(seed, step, rank), images.device)
    if compiled and 'failed' not in _COMPILED:
        try:
            return compiled_pi05_augment()(images, goal, tuple(wrist), params)
        except Exception as error:  # noqa: BLE001 - a compiler failure; a real error repeats in eager below
            _COMPILED['failed'] = repr(error)
            LOGGER.warning('--image-augment pi05: torch.compile failed (%r); the augmentation runs eagerly', error)
    return pi05_augment(images, goal, tuple(wrist), params)
