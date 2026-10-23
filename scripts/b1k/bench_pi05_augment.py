#!/usr/bin/env python3
"""GPU cost of `--image-augment pi05` per rank and step, eager and torch.compile'd (what the trainer uses with
--compile): augment_batch on a per-rank batch of the side runs (256 x 3 cameras + 1 goal view, 224x224) in the
trainer's layout (a uint8 frame-cache batch through prepare_images), timed with CUDA events; its peak extra memory; the
compiled output against eager; and that it never makes the host wait for the GPU, which would stall the trainer's
CPU/GPU overlap: each call is enqueued behind ~0.3 s of GPU work (torch.cuda._sleep) and must return long before that
work finishes (plus torch.cuda.set_sync_debug_mode('error')).

  python scripts/b1k/bench_pi05_augment.py [--batch 256] [--iters 50] [--inductor regions-autotune]

--inductor regions-autotune first sets the global Inductor options the trainer's --compile regions-autotune sets
(max_autotune on, pattern matcher off), which the compiled augmentation then inherits.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_augment import augment_batch, augment_generator, compiled_pi05_augment, pi05_augment, sample_params  # noqa: E402
from b1k_training import WRIST_CAMERAS, prepare_goals, prepare_images  # noqa: E402


def busy_host_ms(call, cycles):
    """Host milliseconds of `call` enqueued behind `cycles` of GPU spinning, and of the spin itself on the GPU."""
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    began = time.perf_counter()
    call()
    host = 1000 * (time.perf_counter() - began)
    torch.cuda.synchronize()
    return host, start.elapsed_time(end)


def measure(images, goal, compiled, iters, warmup):
    for step in range(warmup):
        augment_batch(images, goal, WRIST_CAMERAS, 0, step, 0, compiled)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    torch.cuda.set_sync_debug_mode('error')
    try:
        for index, (start, end) in enumerate(events):
            start.record()
            augment_batch(images, goal, WRIST_CAMERAS, 0, 1000 + index, 0, compiled)
            end.record()
        host = [busy_host_ms(lambda: augment_batch(images, goal, WRIST_CAMERAS, 0, 2000 + i, 0, compiled), 500_000_000)
                for i in range(5)]
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    gpu = sorted(start.elapsed_time(end) for start, end in events)
    return {'gpu_ms_median': round(gpu[len(gpu) // 2], 3), 'gpu_ms_p90': round(gpu[int(0.9 * (len(gpu) - 1))], 3),
            'peak_extra_gib': round((torch.cuda.max_memory_allocated() - base) / 2 ** 30, 2),
            'host_ms_behind_busy_gpu': round(max(h for h, _ in host), 2),
            'busy_gpu_ms': round(min(b for _, b in host), 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--inductor', choices=('default', 'regions-autotune'), default='default')
    args = parser.parse_args()
    if args.inductor == 'regions-autotune':
        import torch._inductor.config as inductor_config
        inductor_config.max_autotune = True
        inductor_config.pattern_matcher = False
    device = torch.device('cuda')
    generator = torch.Generator(device=device).manual_seed(0)
    raw = torch.randint(0, 256, (args.batch, len(WRIST_CAMERAS), *args.size, 3), dtype=torch.uint8, device=device,
                        generator=generator)
    raw_goal = torch.randint(0, 256, (args.batch, 1, *args.size, 3), dtype=torch.uint8, device=device,
                             generator=generator)
    images, goal = prepare_images(raw, True), prepare_goals(raw_goal, True)
    began = time.perf_counter()
    params = sample_params(args.batch, tuple(args.size), augment_generator(0, 7, 0), device)
    fused = compiled_pi05_augment()(images, goal, tuple(WRIST_CAMERAS), params)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - began
    eager = pi05_augment(images, goal, tuple(WRIST_CAMERAS), params)
    difference = max(float((a - b).abs().max()) for a, b in zip(fused, eager))
    print(json.dumps({
        'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__, 'batch': args.batch,
        'size': list(args.size), 'cameras': len(WRIST_CAMERAS), 'goal_views': 1, 'iters': args.iters,
        'inductor': args.inductor,
        'eager': measure(images, goal, False, args.iters, args.warmup),
        'compiled': measure(images, goal, True, args.iters, args.warmup), 'compile_s': round(compile_s, 1),
        'compiled_vs_eager_max_abs': difference,
        'layout_kept': fused[0].stride() == images.stride() and fused[1].stride() == goal.stride()}))


if __name__ == '__main__':
    main()
