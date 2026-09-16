"""Step-based training and standalone checkpoints for upstream ACT and CNNMLP policies."""

import argparse
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from b1k_dataset import B1KDataset, CAMERAS, OBS_KEYS, STATE_INDICES, VIDEO_KEYS, StepBatchSampler
from policy import ACTPolicy, CNNMLPPolicy


LOGGER = logging.getLogger(__name__)
CHECKPOINT_VERSION = 1


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'act-b1k' or checkpoint.get('version') != CHECKPOINT_VERSION:
        raise ValueError('Expected a standalone ACT/CNNMLP B1K checkpoint')
    adapter = checkpoint['adapter_config']
    if adapter['state_indices'] != STATE_INDICES or adapter['action_dim'] != 23:
        raise ValueError('Unsupported checkpoint robot mapping')
    if adapter['video_keys'] != VIDEO_KEYS or adapter['observation_keys'] != OBS_KEYS:
        raise ValueError('Unsupported checkpoint camera mapping')
    if adapter['action_transform'] != 'identity' or adapter['action_shift'] != 0:
        raise ValueError('Unsupported checkpoint action semantics')
    name = policy_class(checkpoint['model_config'])
    if name == 'CNNMLP' and checkpoint['model_config'].get('image_size', [480, 640]) != adapter['image_size']:
        raise ValueError('CNNMLP checkpoint model/adapter image sizes differ')
    return checkpoint


def policy_class(model_config):
    name = model_config.get('policy_class', 'ACT')
    if name not in ('ACT', 'CNNMLP'):
        raise ValueError(f'Unsupported policy class {name}')
    if name == 'CNNMLP' and model_config.get('num_queries', 1) != 1:
        raise ValueError('CNNMLP predicts one action, not an action chunk')
    return name


def make_policy(model_config, device, restoring=False):
    config = dict(model_config, programmatic=True, device=str(device))
    if restoring:
        config['pretrained_backbone'] = False
    policy_type = ACTPolicy if policy_class(config) == 'ACT' else CNNMLPPolicy
    return policy_type(config)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', required=True)
    p.add_argument('--task-names', nargs='+', help='Default: every complete local task')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-steps', type=int, default=50000, help='Total optimizer steps, including resumed steps')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', type=Path)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--policy-class', choices=['ACT', 'CNNMLP'], default='ACT')
    p.add_argument('--chunk-size', type=int, default=100, help='ACT prediction length; CNNMLP always uses one action')
    p.add_argument('--image-size', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'),
                   help='Default: ACT 240 240; CNNMLP 480 640 (original convolution/flatten path)')
    p.add_argument('--position-embedding', choices=['sine', 'learned'], default='sine')
    p.add_argument('--pre-norm', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--hidden-dim', type=int, default=512)
    p.add_argument('--dim-feedforward', type=int, default=3200)
    p.add_argument('--enc-layers', type=int, default=4)
    p.add_argument('--dec-layers', type=int, default=7)
    p.add_argument('--nheads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--kl-weight', type=float, default=10)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lr-backbone', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--pretrained-backbone', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--stats-max-frames', type=int, help='Smoke only: use first N selected frames, not full statistics')
    p.add_argument('--cache-row-groups', type=int, default=2)
    p.add_argument('--timestamp-tolerance', type=float, default=0.008)
    return p


def train(args):
    if min(args.max_steps, args.batch_size, args.save_every) < 1 or args.num_workers < 0:
        raise ValueError('Steps, batch size and save interval must be positive; workers must be nonnegative')
    if args.stats_max_frames is not None and args.stats_max_frames < 2:
        raise ValueError('--stats-max-frames must be at least 2')
    output = Path(args.output_dir).resolve()
    root = Path(args.dataset_path).resolve()
    if output == root or root in output.parents:
        raise ValueError('--output-dir cannot be inside the read-only dataset')
    if not args.resume and ((output / 'run.json').exists() or any(output.glob('step_*.pt'))):
        raise FileExistsError(f'{output} already contains a run; use --resume or another output directory')
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        if args.max_steps <= checkpoint['step']:
            raise ValueError(f'--max-steps must exceed resumed step {checkpoint["step"]}')
        conflicts = [p for p in output.glob('step_*.pt')
                     if p.stem[5:].isdigit() and checkpoint['step'] < int(p.stem[5:]) <= args.max_steps]
        if conflicts:
            raise FileExistsError(f'Resume would overwrite existing snapshots: {conflicts}; choose another output directory')
        LOGGER.info('Resuming %s at optimizer step %d; saved architecture/preprocessing/seed are authoritative',
                    args.resume, checkpoint['step'])
        adapter = checkpoint['adapter_config']
        model_config = dict(checkpoint['model_config'])
        model_config.setdefault('policy_class', 'ACT')
        task_names = args.task_names or list(checkpoint['task_map'].values())
        args.seed = checkpoint['train_config']['seed']
    else:
        image_size = args.image_size or ([240, 240] if args.policy_class == 'ACT' else [480, 640])
        if args.policy_class == 'CNNMLP' and (args.pre_norm or args.position_embedding != 'sine'):
            raise ValueError('Position embeddings and pre-norm are ACT architecture options; CNNMLP does not use them')
        if min(image_size) < 1:
            raise ValueError('--image-size dimensions must be positive')
        if args.policy_class == 'CNNMLP' and min(image_size) < 385:
            raise ValueError('CNNMLP requires images at least 385x385; default is 480x640')
        adapter = {'image_size': list(image_size), 'state_indices': STATE_INDICES,
                   'video_keys': VIDEO_KEYS, 'observation_keys': OBS_KEYS, 'action_dim': 23,
                   'action_transform': 'identity', 'action_shift': 0, 'task_conditioning': 'onehot',
                   'timestamp_tolerance': args.timestamp_tolerance, 'image_resize': 'bilinear_antialias',
                   'image_normalization': f'rgb_div255_then_imagenet_in_{args.policy_class}Policy'}
        task_names = args.task_names
        model_config = {'policy_class': args.policy_class,
                        'num_queries': args.chunk_size if args.policy_class == 'ACT' else 1,
                        'hidden_dim': args.hidden_dim,
                        'dim_feedforward': args.dim_feedforward, 'enc_layers': args.enc_layers,
                        'dec_layers': args.dec_layers, 'nheads': args.nheads, 'dropout': args.dropout,
                        'kl_weight': args.kl_weight, 'lr': args.lr, 'lr_backbone': args.lr_backbone,
                        'weight_decay': args.weight_decay, 'backbone': 'resnet18',
                        'pretrained_backbone': args.pretrained_backbone, 'camera_names': CAMERAS,
                        'position_embedding': args.position_embedding, 'pre_norm': args.pre_norm,
                        'dilation': False, 'masks': False, 'action_dim': 23}
        if args.policy_class == 'CNNMLP':
            model_config['image_size'] = list(image_size)
    if policy_class(model_config) == 'ACT':
        if model_config['hidden_dim'] % 4 or model_config['hidden_dim'] % model_config['nheads']:
            raise ValueError('hidden-dim must be divisible by four and nheads')
    dataset = B1KDataset(root, task_names, model_config['num_queries'], adapter['image_size'],
                         cache_row_groups=args.cache_row_groups,
                         timestamp_tolerance=adapter['timestamp_tolerance'])
    if checkpoint and dataset.task_map != checkpoint['task_map']:
        raise ValueError('Resume task map differs from checkpoint')
    model_config['state_dim'] = len(STATE_INDICES) + len(dataset.task_map)
    fingerprint = dataset.fingerprint()
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        stats = checkpoint['normalization']
        if stats['fingerprint'] != fingerprint:
            raise ValueError('Resume dataset metadata/files differ from checkpoint; use the same local subset')
    else:
        stats_path = output / f'stats_{fingerprint[:16]}_{args.stats_max_frames or "all"}.json'
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
        else:
            stats = dataset.compute_stats(args.stats_max_frames)
            stats_path.write_text(json.dumps(stats, indent=2))
    if stats['approximate']:
        LOGGER.warning('SMOKE statistics only: %d/%d frames; not full-dataset normalization', stats['count'], len(dataset))
    dataset.stats = stats
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    policy = make_policy(model_config, device, restoring=checkpoint is not None)
    optimizer = policy.configure_optimizers()
    start = 0
    if checkpoint:
        policy.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['step']
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state(checkpoint['cuda_rng'], device)
    if args.max_steps <= start:
        raise ValueError(f'--max-steps must exceed resumed step {start}')
    train_config = vars(args).copy()
    train_config['resume'] = str(args.resume) if args.resume else None
    run = {'model_config': model_config, 'adapter_config': adapter, 'train_config': train_config,
           'task_map': dataset.task_map, 'normalization': stats}
    (output / ('resume_run.json' if checkpoint else 'run.json')).write_text(json.dumps(run, indent=2))
    sampler = StepBatchSampler(dataset, args.batch_size, start, args.max_steps, args.seed)
    loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': device.type == 'cuda',
                     'generator': torch.Generator().manual_seed(args.seed)}
    if args.num_workers:
        loader_kwargs.update(multiprocessing_context='spawn', prefetch_factor=1)
    loader = DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)
    policy.train()
    LOGGER.info('Training %d -> %d steps on %s with real upstream %s',
                start, args.max_steps, device, policy_class(model_config))
    begin = time.monotonic()
    try:
        with (output / 'metrics.jsonl').open('a') as metrics:
            for step, batch in enumerate(loader, start + 1):
                images, qpos, actions, is_pad = [x.to(device, non_blocking=True) for x in batch]
                optimizer.zero_grad(set_to_none=True)
                losses = policy(qpos, images, actions, is_pad)
                if not torch.isfinite(losses['loss']):
                    raise FloatingPointError(f'Non-finite loss at step {step}')
                losses['loss'].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'), error_if_nonfinite=True)
                optimizer.step()
                record = {'step': step, **{k: float(v.detach()) for k, v in losses.items()},
                          'grad_norm': float(grad_norm), 'elapsed_s': time.monotonic() - begin}
                metrics.write(json.dumps(record) + '\n')
                metrics.flush()
                LOGGER.info('%s', json.dumps(record))
                if step % args.save_every == 0 or step == args.max_steps:
                    path = output / f'step_{step:08d}.pt'
                    if path.exists():
                        raise FileExistsError(f'Refusing to overwrite checkpoint {path}')
                    saved = {'format': 'act-b1k', 'version': CHECKPOINT_VERSION, 'step': step,
                             **run, 'model': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                             'torch_rng': torch.get_rng_state(),
                             'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
                    temporary = path.with_suffix('.tmp')
                    torch.save(saved, temporary)
                    temporary.replace(path)
                    latest = output / 'latest.pt'
                    link = output / 'latest.tmp'
                    link.unlink(missing_ok=True)
                    link.symlink_to(path.name)
                    link.replace(latest)
                    LOGGER.info('Saved standalone checkpoint %s', path)
    finally:
        dataset.close()
    return path


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    train(parser().parse_args())
