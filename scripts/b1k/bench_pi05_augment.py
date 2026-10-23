#!/usr/bin/env python3
"""GPU cost of `--image-augment pi05` per rank and step: augment_batch on a per-rank batch of the side runs (256 x 3
cameras + 1 goal view, 224x224) in the trainer's layout (a uint8 frame-cache batch through prepare_images), timed with
CUDA events, its peak extra memory, and a check that it never makes the host wait for the GPU
(torch.cuda.set_sync_debug_mode('error')), which would stall the trainer's CPU/GPU overlap.

  python scripts/b1k/bench_pi05_augment.py [--batch 256] [--iters 50]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_augment import augment_batch  # noqa: E402
from b1k_training import WRIST_CAMERAS, prepare_goals, prepare_images  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--iters', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=5)
    args = parser.parse_args()
    device = torch.device('cuda')
    generator = torch.Generator(device=device).manual_seed(0)
    raw = torch.randint(0, 256, (args.batch, len(WRIST_CAMERAS), *args.size, 3), dtype=torch.uint8, device=device,
                        generator=generator)
    raw_goal = torch.randint(0, 256, (args.batch, 1, *args.size, 3), dtype=torch.uint8, device=device,
                             generator=generator)
    images, goal = prepare_images(raw, True), prepare_goals(raw_goal, True)
    for step in range(args.warmup):
        augment_batch(images, goal, WRIST_CAMERAS, 0, step, 0, True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.iters)]
    host = []
    torch.cuda.set_sync_debug_mode('error')
    try:
        for index, (start, end) in enumerate(events):
            began = time.perf_counter()
            start.record()
            out, out_goal = augment_batch(images, goal, WRIST_CAMERAS, 0, 1000 + index, 0, True)
            end.record()
            host.append(1000 * (time.perf_counter() - began))
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    gpu = sorted(start.elapsed_time(end) for start, end in events)
    host.sort()
    layout = out.permute(1, 0, 3, 4, 2).is_contiguous() and out_goal.permute(1, 0, 3, 4, 2).is_contiguous()
    print(json.dumps({
        'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__, 'batch': args.batch,
        'size': list(args.size), 'cameras': len(WRIST_CAMERAS), 'goal_views': 1, 'iters': args.iters,
        'gpu_ms_median': round(gpu[len(gpu) // 2], 3), 'gpu_ms_p90': round(gpu[int(0.9 * (len(gpu) - 1))], 3),
        'host_ms_median': round(host[len(host) // 2], 3), 'peak_extra_gib': round(
            (torch.cuda.max_memory_allocated() - base) / 2 ** 30, 2),
        'no_host_sync': True, 'trainer_layout': layout}))


if __name__ == '__main__':
    main()
