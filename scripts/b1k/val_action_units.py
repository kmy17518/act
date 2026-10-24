#!/usr/bin/env python3
"""Validation metrics of a checkpoint on the frames its run validates on, val/l1_action_units included.

val/l1_action_units is the zero-latent L1 between predicted and recorded actions, both absolute and in action units,
so runs with different action representations or normalizations compare (b1k_training.action_unit_errors). Runs from
before the metric (e.g. TODO 1's, absolute z-score) are evaluated here on the same frames as the side runs: the
checkpoint's validation episodes, batches StepBatchSampler(val episodes, val batch size, 0, val batches, seed) as in
training, its own statistics, preprocessing and autocast. --settle-steps validates on another settle cut of the same
validation episodes: with 0, a run trained with settle frames is measured on exactly the frames of runs trained
without them (same episodes, seed and batches). Takes full or eval-only checkpoints, e.g.

    hf download kmy17518/act-isg1020-multitask5-bs1024-s0 eval/step-00100000.pt --local-dir /tmp/dev/runs/todo1-eval
    python scripts/b1k/val_action_units.py --checkpoint /tmp/dev/runs/todo1-eval/eval/step-00100000.pt \\
        --dataset-path /tmp/dev/datasets/isg-1020-b1k --frame-cache /tmp/dev/datasets/isg-1020-b1k-act-frame-cache-224x224
"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from b1k_dataset import B1KDataset, parse_settle_steps  # noqa: E402
from b1k_language import language_embedding_table  # noqa: E402
from b1k_training import evaluate_validation, goal_config, load_checkpoint, make_policy  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset-path', type=Path, help="Default: the run's dataset path")
    p.add_argument('--frame-cache', type=Path, help="Default: the run's frame cache when it exists, else video decoding")
    p.add_argument('--device', default='cuda')
    p.add_argument('--val-batches', type=int, help="Default: the run's (16)")
    p.add_argument('--val-batch-size', type=int, help="Default: the run's (min(64, batch size))")
    p.add_argument('--settle-steps', metavar='SPEC',
                   help="Validate on this --settle-steps cut of the run's validation episodes (all, a number of frames "
                        "or a fraction of the program length, e.g. 0) instead of the run's own. Default: the run's")
    p.add_argument('--output', type=Path, help='Also write the metrics as JSON')
    args = p.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    if not checkpoint.get('val_split'):
        raise SystemExit(f'{args.checkpoint} was trained without a validation split')
    config, adapter, train = checkpoint['model_config'], checkpoint['adapter_config'], checkpoint['train_config']
    root = (args.dataset_path or Path(train['dataset_path'])).resolve()
    cache = args.frame_cache or (Path(train['frame_cache']) if train.get('frame_cache') else None)
    cache = cache.resolve() if cache is not None and cache.exists() else None
    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    goal = goal_config(config)
    run_settle = (checkpoint.get('settle_steps') or {}).get('spec', 'all')
    settle = run_settle if args.settle_steps is None else parse_settle_steps(args.settle_steps)
    dataset = B1KDataset(root, list(checkpoint['task_map'].values()), config['num_queries'], adapter['image_size'],
                         timestamp_tolerance=adapter['timestamp_tolerance'], frame_cache=cache,
                         goal_views=goal['goal_views'], goal_source=adapter.get('goal_source', 'episode_last'),
                         task_onehot=adapter.get('task_conditioning', 'onehot') == 'onehot',
                         episodes=checkpoint['val_split']['val_episodes'],
                         settle_steps=settle,
                         gripper_state=adapter.get('gripper_state', 'fingers'),
                         goal_mismatch_views=adapter.get('goal_mismatch_views', []),
                         goal_mismatch_seed=train['seed'])
    if dataset.task_map != checkpoint['task_map']:
        raise SystemExit('The dataset task map differs from the checkpoint')
    dataset.stats = checkpoint['normalization']
    torch.set_float32_matmul_precision(train.get('matmul_precision', 'highest'))
    autocast_mode = train.get('autocast', 'none') if cuda else 'none'
    policy = make_policy(config, device, restoring=True,
                         backbone_autocast_dtype=torch.bfloat16 if autocast_mode == 'bf16-backbone' else None,
                         fast_maxpool=bool(train.get('fast_maxpool', True)) and cuda,
                         fast_stem=bool(train.get('fast_stem', True)) and cuda)
    channels_last = bool(train.get('channels_last', True))
    if channels_last:
        policy.to(memory_format=torch.channels_last)
    policy.load_state_dict(checkpoint['model'])
    embeddings = language_embedding_table(config, checkpoint['task_map'], checkpoint.get('language_cache'))
    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if autocast_mode == 'bf16' else nullcontext
    metrics = evaluate_validation(policy, dataset, args.val_batches or train.get('val_batches', 16),
                                  args.val_batch_size or train.get('val_batch_size') or min(64, train['batch_size']),
                                  None if embeddings is None else embeddings.to(device), checkpoint['task_map'],
                                  device, autocast, channels_last, train['seed'])
    val_frames = len(dataset)
    dataset.close()
    result = {'checkpoint': str(args.checkpoint), 'step': checkpoint['step'],
              'action_representation': config.get('action_representation', 'absolute'),
              'normalization': config.get('normalization', 'zscore'), 'settle_steps': settle,
              'run_settle_steps': run_settle, 'val_frames': val_frames, 'metrics': metrics}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
