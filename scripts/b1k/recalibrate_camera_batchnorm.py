#!/usr/bin/env python3
"""Convert a `--backbone-norm batch` ACT checkpoint (one backbone pass per camera) to `batch_per_camera`.

Such a checkpoint was trained normalizing every camera pass by that camera's own batch statistics, but it
carries one running mean/variance per layer -- a mixture over the cameras -- so eval mode normalizes each
camera differently from training (b1k_runs.md). This tool loads the weights into the per-camera BatchNorm
layout, re-estimates one set of running statistics per camera as the cumulative average over training
batches (BatchNorm in train mode, everything else in eval mode, no gradients), and writes a checkpoint of
the same type (full checkpoints stay resumable: parameters and optimizer state are untouched). With
--evaluate it also reports the teacher-forced L1 on the calibration batches before and after.
"""

import argparse
from functools import partial
import json
import logging
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_dataset import B1KDataset, SplitBatchSampler, StepBatchSampler  # noqa: E402
from b1k_language import language_embedding_table, language_for_qpos  # noqa: E402
from b1k_training import (CHECKPOINT_VERSION, atomic_save, configure_cpu_threads, consume_batch, load_checkpoint,  # noqa: E402
                          make_policy, optimizer_batches, policy_class)
from detr.models.backbone import PerCameraBatchNorm2d  # noqa: E402

LOGGER = logging.getLogger('recalibrate_camera_batchnorm')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True, help='Full or eval checkpoint with backbone_norm batch')
    p.add_argument('--output', type=Path, required=True, help='New checkpoint path (must not exist)')
    p.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', type=Path, required=True)
    p.add_argument('--frame-cache', type=Path)
    p.add_argument('--batches', type=int, default=16, help='Calibration batches (cumulative average)')
    p.add_argument('--batch-size', type=int, default=1560)
    p.add_argument('--loader-batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--prefetch-factor', type=int, default=2)
    p.add_argument('--seed', type=int, default=None, help='Sampler seed; default: the checkpoint training seed')
    p.add_argument('--start-step', type=int, default=None,
                   help='Sampler position; default: the checkpoint step (the batches training would see next)')
    p.add_argument('--device', default='cuda')
    p.add_argument('--matmul-precision', choices=['highest', 'high', 'medium'], default='high')
    p.add_argument('--channels-last', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--evaluate', action='store_true', help='Report teacher-forced L1 before/after on the same batches')
    return p


def per_camera_layers(policy):
    layers = [module for module in policy.modules() if isinstance(module, PerCameraBatchNorm2d)]
    if not layers:
        raise ValueError('The converted policy has no per-camera BatchNorm layers')
    return layers


def batch_iterator(dataset, args, device, seed, start):
    sampler = StepBatchSampler(dataset, args.batch_size, start, start + args.batches, seed)
    loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': device.type == 'cuda',
                     'generator': torch.Generator().manual_seed(seed)}
    if args.num_workers:
        loader_kwargs.update(multiprocessing_context='spawn', prefetch_factor=args.prefetch_factor,
                             worker_init_fn=partial(configure_cpu_threads, torch_threads=1, arrow_threads=1,
                                                    opencv_threads=1))
    loader_sampler = SplitBatchSampler(sampler, args.loader_batch_size) if args.loader_batch_size else sampler
    loader = DataLoader(dataset, batch_sampler=loader_sampler, **loader_kwargs)
    for batch in optimizer_batches(loader, args.batch_size, args.loader_batch_size, device=device):
        images, qpos, actions, is_pad = consume_batch(batch, None, args.channels_last)[:4]
        yield images, qpos, actions, is_pad


@torch.no_grad()
def teacher_forced_l1(policy, batches, embeddings, seed=1234):
    """Mean L1 with the style latent sampled identically for every call (eval mode, no dropout)."""
    total = count = 0.0
    for images, qpos, actions, is_pad in batches:
        torch.manual_seed(seed)
        losses = policy(qpos, images, actions, is_pad, lang_emb=language_for_qpos(qpos, embeddings))
        total += float(losses['l1'])
        count += 1
    return total / count


@torch.no_grad()
def recalibrate(policy, batches, embeddings):
    """Cumulative per-camera running statistics from the current weights; returns the number of batches used."""
    layers = per_camera_layers(policy)
    momenta = [layer.momentum for layer in layers]
    policy.eval()
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    count = 0
    for images, qpos, _, _ in batches:
        policy(qpos, images, lang_emb=language_for_qpos(qpos, embeddings))  # one pass per camera, statistics per camera
        count += 1
    for layer, momentum in zip(layers, momenta):
        layer.momentum = momentum
        layer.eval()
    if count == 0:
        raise ValueError('No calibration batches')
    tracked = {int(getattr(layer, f'num_batches_tracked_{camera}')) for layer in layers for camera in range(layers[0].num_cameras)}
    if tracked != {count}:
        raise RuntimeError(f'Expected every camera statistic to see {count} batches, saw {sorted(tracked)}')
    return count


def convert(args):
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists')
    if args.batches < 1 or args.batch_size < 1:
        raise ValueError('--batches and --batch-size must be positive')
    checkpoint = load_checkpoint(args.checkpoint)
    model_config = dict(checkpoint['model_config'])
    if policy_class(model_config) != 'ACT':
        raise ValueError('Per-camera BatchNorm statistics apply to the shared ACT backbone')
    if model_config.get('backbone_norm', 'frozen') != 'batch' or model_config.get('camera_batch', False):
        raise ValueError('Expected a backbone_norm=batch checkpoint trained with one backbone pass per camera')
    model_config['backbone_norm'] = 'batch_per_camera'
    device = torch.device(args.device)
    torch.set_float32_matmul_precision(args.matmul_precision)
    configure_cpu_threads(torch_threads=torch.get_num_threads(), arrow_threads=1, opencv_threads=1)
    policy = make_policy(model_config, device, restoring=True)
    policy.load_state_dict(checkpoint['model'])  # single statistics are copied to every camera
    if args.channels_last:
        policy.to(memory_format=torch.channels_last)
    policy.eval()
    embeddings = language_embedding_table(model_config, checkpoint['task_map'], checkpoint.get('language_cache'))
    embeddings = embeddings.to(device) if embeddings is not None else None
    adapter = checkpoint['adapter_config']
    dataset = B1KDataset(args.dataset_path.resolve(), list(checkpoint['task_map'].values()), model_config['num_queries'],
                         adapter['image_size'], profile_reads=True, timestamp_tolerance=adapter['timestamp_tolerance'],
                         frame_cache=args.frame_cache.resolve() if args.frame_cache else None)
    try:
        if dataset.task_map != checkpoint['task_map']:
            raise ValueError('Dataset task map differs from the checkpoint')
        dataset.stats = checkpoint['normalization']
        fingerprint = dataset.fingerprint()
        if fingerprint != checkpoint['normalization']['fingerprint']:
            LOGGER.warning('Dataset fingerprint differs from the checkpoint (%s vs %s); statistics are estimated on '
                           'these files anyway', fingerprint[:16], checkpoint['normalization']['fingerprint'][:16])
        seed = checkpoint['train_config']['seed'] if args.seed is None else args.seed
        start = checkpoint['step'] if args.start_step is None else args.start_step
        batches = lambda: batch_iterator(dataset, args, device, seed, start)  # noqa: E731
        report = {'source': str(args.checkpoint.resolve()), 'batches': args.batches, 'batch_size': args.batch_size,
                  'seed': seed, 'start_step': start, 'dataset_fingerprint': fingerprint,
                  'method': 'cumulative average of per-camera-pass batch statistics, BatchNorm train mode only'}
        if args.evaluate:
            before = time.monotonic()
            report['l1_mixed_running_statistics'] = teacher_forced_l1(policy, batches(), embeddings)
            LOGGER.info('L1 with the mixed running statistics (eval mode as trained): %.5f (%.0f s)',
                        report['l1_mixed_running_statistics'], time.monotonic() - before)
        before = time.monotonic()
        used = recalibrate(policy, batches(), embeddings)
        LOGGER.info('Re-estimated per-camera statistics over %d batches of %d (%.0f s)', used, args.batch_size,
                    time.monotonic() - before)
        if args.evaluate:
            report['l1_per_camera_running_statistics'] = teacher_forced_l1(policy, batches(), embeddings)
            LOGGER.info('L1 with per-camera running statistics (eval mode): %.5f', report['l1_per_camera_running_statistics'])
    finally:
        dataset.close()
    saved = {key: value for key, value in checkpoint.items() if key != 'model'}
    saved.update({'format': 'act-b1k', 'version': CHECKPOINT_VERSION, 'model_config': model_config,
                  'model': {key: value.detach().cpu() for key, value in policy.state_dict().items()},
                  'camera_batchnorm_recalibration': report})
    atomic_save(saved, args.output)
    load_checkpoint(args.output)
    LOGGER.info('Wrote %s checkpoint %s', saved.get('checkpoint_type', 'full'), args.output)
    return report


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    report = convert(parser().parse_args())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
