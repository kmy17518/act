#!/usr/bin/env python3
"""Offline goal-sensitivity diagnostic for a goal-conditioned ACT checkpoint (plan section 12, quick check only).

For sampled training frames it predicts the action chunk with (a) the episode's own goal image, (b) the goal image
of an episode of the other task, and (c) the goal masked out (late fusion), and reports the normalized L1 to the
frame's recorded actions plus the maximum prediction change. Own < other shows the policy reads the goal; it is
NOT goal-following evidence (no rollout, training frames, zero inference latent). Language-conditioned
checkpoints get their prompt from the frame's own task in every case (language held constant, goal swapped).

    goal_sensitivity_probe.py CHECKPOINT --dataset-path ROOT [--frame-cache DIR] [--samples 8] [--seed 0]
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_dataset import B1KDataset  # noqa: E402
from b1k_language import language_embedding_table, language_for_tasks  # noqa: E402
from b1k_training import consume_batch, goal_config, load_checkpoint, make_policy  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint')
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--frame-cache')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = load_checkpoint(args.checkpoint)
    config, adapter = checkpoint['model_config'], checkpoint['adapter_config']
    goal = goal_config(config)
    if goal['goal_fusion'] == 'none':
        raise SystemExit('Not a goal-conditioned checkpoint')
    print('conditioning:', {k: v for k, v in checkpoint.get('conditioning', {}).items() if k in ('regime', 'source_commit')},
          '| goal', goal['goal_fusion'], goal['goal_views'], '| language', config.get('language_conditioning', 'none'))
    device = torch.device(args.device)
    policy = make_policy(config, device, restoring=True)
    policy.load_state_dict(checkpoint['model'])
    policy.eval()
    embeddings = language_embedding_table(config, checkpoint['task_map'], checkpoint.get('language_cache'))
    embeddings = embeddings.to(device) if embeddings is not None else None
    dataset = B1KDataset(args.dataset_path, list(checkpoint['task_map'].values()), config['num_queries'], adapter['image_size'],
                         goal_views=goal['goal_views'], goal_source=adapter.get('goal_source', 'episode_last'),
                         task_onehot=adapter.get('task_conditioning', 'onehot') == 'onehot',
                         frame_cache=Path(args.frame_cache).resolve() if args.frame_cache else None)
    if dataset.task_map != checkpoint['task_map']:
        raise SystemExit(f'Dataset task map {dataset.task_map} differs from the checkpoint {checkpoint["task_map"]}')
    dataset.stats = checkpoint['normalization']
    rng = np.random.default_rng(args.seed)
    by_task = {}
    for position, ep in enumerate(dataset.episodes):
        by_task.setdefault(int(ep['task_index']), []).append(position)
    rows = []
    with torch.no_grad():
        for position in rng.choice(len(dataset.episodes), min(args.samples, len(dataset.episodes)), replace=False):
            ep = dataset.episodes[position]
            task = int(ep['task_index'])
            other_tasks = [t for t in by_task if t != task] or [task]
            other_position = int(rng.choice(by_task[int(rng.choice(other_tasks))]))
            frame = int(rng.integers(0, max(1, ep['length'] // 2)))
            images, qpos, actions, is_pad, own_goal, task_id, _ = consume_batch(
                tuple(x[None] for x in dataset.sample_at(ep['episode_index'], frame)) + ({},))
            other_goal = consume_batch(tuple(x[None] for x in dataset.sample_at(
                dataset.episodes[other_position]['episode_index'], 0)) + ({},))[4]
            kwargs = {}
            if embeddings is not None:
                kwargs['lang_emb'] = language_for_tasks(task_id.to(device), embeddings, checkpoint['task_map'])
            to = lambda t: t.to(device)  # noqa: E731
            pred_own = policy(to(qpos), to(images), goal=to(own_goal), **kwargs)[0].cpu()
            pred_other = policy(to(qpos), to(images), goal=to(other_goal), **kwargs)[0].cpu()
            masked = None
            if goal['goal_fusion'] == 'late':
                masked = policy(to(qpos), to(images), goal=to(own_goal), goal_valid=torch.zeros(1, len(goal['goal_views']), dtype=torch.bool, device=device),
                                **kwargs)[0].cpu()
            valid = ~is_pad[0]
            target = actions[0]
            l1 = lambda p: (p - target).abs()[valid].mean().item()  # noqa: E731
            rows.append((int(ep['episode_index']), task, frame, l1(pred_own), l1(pred_other),
                         l1(masked) if masked is not None else float('nan'), (pred_own - pred_other).abs().max().item()))
    for r in rows:
        print('episode %4d task %d frame %4d | L1 own goal %.4f | other-task goal %.4f | goal masked %.4f | max |dpred| %.3f' % r)
    masked_mean = np.mean([r[5] for r in rows]) if goal['goal_fusion'] == 'late' else float('nan')
    print('mean L1: own goal %.4f | other-task goal %.4f | goal masked %.4f (%d samples; normalized action units)' % (
        np.mean([r[3] for r in rows]), np.mean([r[4] for r in rows]), masked_mean, len(rows)))


if __name__ == '__main__':
    main()
