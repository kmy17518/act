#!/usr/bin/env python3
"""Tier A offline goal metrics of a goal-conditioned ACT checkpoint (docs/isg-goal-conditioning-plan.md section 4).

For N frames per task, stratified over episodes and time, of the held-out episodes (the split file's per-task
`held_out_eval` lists of an eval split, else `held_out` of the training split) and of the training episodes (the
checkpoint's episode split), the policy predicts its action chunk (eval mode, zero inference latent) with
  (a) the episode's own goal image,
  (b) the goal image of another episode of the same task and subset (drawn per frame, fixed by the seed),
  (c) the goal masked (goal_valid=False: late fusion's key-padding mask, early fusion's zero goal half),
  (d) every goal view and its camera's current image trading places,
and reports, in normalized action units:
  L1_own     masked L1 of (a) to the recorded actions (mean over the valid chunk steps and the 23 dimensions)
  L1_other   the same for (b);  gap = L1_other - L1_own  (does the policy read goal *content*)
  L1_masked  the same for (c)   (how much it depends on the goal at all)
  swap       mean over frames of the largest |(d) - (a)| over the valid chunk (does it know which image is the goal)
Frame selection, the other-goal draw and the subsets are deterministic in the seed, so checkpoints compare paired.

A checkpoint trained with `--goal-mismatch-views` (the wrist-goal control) gets the same rule here: those views show
the goal images of another episode of the same task and subset, drawn per frame from a separate stream of the seed
(frames and the other-goal draw stay those of every other checkpoint), in both (a) and (b), so the gap measures the
reliance on the remaining (head) goal. `--goal-mismatch-views` applies the same rule at evaluation only to a checkpoint
trained on its own goals (a model trained on real wrist goals may also suffer from the distribution shift).

`--heldout-dataset ROOT` takes the held-out episodes from a separate LeRobot root (the test set kmy17518/isg-init-eval,
collected on instances no training demo uses), listed by its `isg_meta/eval_split.json` (or `--split`). Tasks are
matched by name (the root has its own task indices), its frames are decoded from video and rounded to uint8 exactly
like the training frame cache, and the training subset still comes from the checkpoint's dataset root and frame cache.

    isg_tier_a.py CHECKPOINT [--dataset-path ROOT] [--frame-cache DIR] [--heldout-dataset ROOT] [--split FILE]
                  [--frames-per-task 256] [--seed 0] [--device cuda] [--batch-size 64] [--output JSON]
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_dataset import CAMERAS, GOAL_MISMATCH_STREAM, B1KDataset, quantize_image  # noqa: E402
from b1k_language import language_embedding_table, language_for_tasks  # noqa: E402
from b1k_training import goal_config, load_checkpoint, make_policy, prepare_goals, prepare_images  # noqa: E402

FORMAT = 'isg-tier-a/v1'
METRICS = ('L1_own', 'L1_other', 'gap', 'L1_masked', 'swap', 'swap_mean')


def held_out_lists(split_path):
    split = json.loads(Path(split_path).read_text())
    result = {}
    for task, entry in split['tasks'].items():
        episodes = entry.get('held_out_eval', entry.get('held_out', []))
        if isinstance(episodes, list) and episodes:
            result[task] = sorted(int(e) for e in episodes)
    return result, split.get('name')


def stratified_frames(dataset, episodes, count, rng):
    """`count` (episode_index, frame) pairs: episodes as evenly as possible, frames stratified over each episode's
    kept length (one uniform draw per equal-width time stratum)."""
    episodes = list(rng.permutation(episodes))
    per_episode = [count // len(episodes) + (i < count % len(episodes)) for i in range(len(episodes))]
    frames = []
    for episode, n in zip(episodes, per_episode):
        length = int(dataset.by_id[int(episode)]['length'])
        for i in range(n):
            frames.append((int(episode), min(length - 1, int((i + rng.random()) * length / n))))
    return frames


def other_episodes(frames, episodes, rng):
    choices = {}
    for episode in sorted(set(e for e, _ in frames)):
        others = [e for e in episodes if e != episode]
        choices[episode] = others
    return [int(rng.choice(choices[episode])) if choices[episode] else None for episode, _ in frames]


class Probe:
    def __init__(self, checkpoint, device, batch_size, mismatch_views=None):
        self.config, self.adapter = checkpoint['model_config'], checkpoint['adapter_config']
        self.goal = goal_config(self.config)
        if self.goal['goal_fusion'] == 'none':
            raise SystemExit('Not a goal-conditioned checkpoint')
        self.device, self.batch_size = torch.device(device), batch_size
        self.policy = make_policy(self.config, self.device, restoring=True)
        self.policy.load_state_dict(checkpoint['model'])
        self.policy.eval()
        self.task_map = checkpoint['task_map']
        embeddings = language_embedding_table(self.config, self.task_map, checkpoint.get('language_cache'))
        self.embeddings = embeddings.to(self.device) if embeddings is not None else None
        self.cameras = [CAMERAS.index(view) for view in self.goal['goal_views']]
        trained = list(self.adapter.get('goal_mismatch_views', []))
        self.mismatch_evaluation_only = bool(mismatch_views) and not trained
        if mismatch_views and trained and list(mismatch_views) != trained:
            raise SystemExit(f'The checkpoint trained with --goal-mismatch-views {trained}; Tier A applies exactly those')
        self.mismatch_views = list(mismatch_views or trained)
        if not set(self.mismatch_views) <= set(self.goal['goal_views']) or len(self.mismatch_views) == len(self.goal['goal_views']):
            raise SystemExit(f'--goal-mismatch-views {self.mismatch_views} must be some, not all, of the goal views '
                             f'{self.goal["goal_views"]}')

    def predict(self, qpos, images, goal, task_id, goal_valid=None):
        kwargs = {}
        if self.embeddings is not None:
            kwargs['lang_emb'] = language_for_tasks(task_id, self.embeddings, self.task_map)
        return self.policy(qpos, images, goal=goal, goal_valid=goal_valid, **kwargs).float()

    def checkpoint_task_ids(self, dataset):
        """Dataset task index -> checkpoint task index, by task name (a separate held-out root has its own indices)."""
        by_name = {name: index for index, name in self.task_map.items()}
        return {index: by_name[name] for index, name in dataset.task_map.items()}

    @staticmethod
    def frames_uint8(images):
        """Uncached float (camera, 3, H, W) frames to the uint8 (camera, H, W, 3) frames the training cache holds."""
        if images.dtype == torch.uint8:
            return images
        return torch.from_numpy(np.stack([quantize_image(image) for image in images]))

    @torch.no_grad()
    def evaluate(self, dataset, frames, others, partners=None):
        """Per-frame metrics; `partners` (goal-mismatch checkpoints) supply the mismatched views' goals for (a) and (b)."""
        rows = []
        task_ids = self.checkpoint_task_ids(dataset)
        positions = dataset.positions
        for start in range(0, len(frames), self.batch_size):
            chunk = list(zip(frames[start:start + self.batch_size], others[start:start + self.batch_size]))
            samples = [dataset.sample_at(episode, frame) for (episode, frame), _ in chunk]
            other_goals = [dataset.goal_for(positions[other if other is not None else episode])
                           for (episode, _), other in chunk]
            if partners is not None:
                mates = [positions[partner] for partner in partners[start:start + self.batch_size]]
                own_goals = [dataset.goal_with_partner(positions[episode], mate) for ((episode, _), _), mate in zip(chunk, mates)]
                other_goals = [dataset.goal_with_partner(positions[other if other is not None else episode], mate)
                               for ((episode, _), other), mate in zip(chunk, mates)]
                samples = [(*s[:4], goal, s[5]) for s, goal in zip(samples, own_goals)]
            images = prepare_images(torch.stack([self.frames_uint8(s[0]) for s in samples]).to(self.device))
            qpos = torch.stack([torch.as_tensor(s[1]) for s in samples]).float().to(self.device)
            actions = torch.stack([s[2] for s in samples]).to(self.device)[:, :self.policy.model.num_queries]
            valid = ~torch.stack([s[3] for s in samples]).to(self.device)[:, :self.policy.model.num_queries]
            goal = prepare_goals(torch.stack([s[4] for s in samples]).to(self.device))
            other = prepare_goals(torch.stack(other_goals).to(self.device))
            task_id = torch.tensor([task_ids[int(s[5])] for s in samples], dtype=torch.int64, device=self.device)
            own = self.predict(qpos, images, goal, task_id)
            swapped_images, swapped_goal = images.clone(), goal.clone()
            for view, camera in enumerate(self.cameras):
                swapped_images[:, camera] = goal[:, view]
                swapped_goal[:, view] = images[:, camera]
            predictions = {
                'own': own, 'other': self.predict(qpos, images, other, task_id),
                'masked': self.predict(qpos, images, goal, task_id,
                                       torch.zeros(len(chunk), len(self.cameras), dtype=torch.bool, device=self.device)),
                'swap': self.predict(qpos, swapped_images, swapped_goal, task_id)}
            mask = valid[..., None].float()
            count = mask.sum(dim=(1, 2)) * actions.shape[-1]

            def l1(prediction):
                return ((prediction - actions).abs() * mask).sum(dim=(1, 2)) / count

            change = (predictions['swap'] - own).abs() * mask
            per_frame = {'L1_own': l1(own), 'L1_other': l1(predictions['other']), 'L1_masked': l1(predictions['masked']),
                         'swap': change.flatten(1).max(dim=1).values, 'swap_mean': change.sum(dim=(1, 2)) / count}
            per_frame['gap'] = per_frame['L1_other'] - per_frame['L1_own']
            has_other = torch.tensor([other is not None for _, other in chunk], device=self.device)
            for i in range(len(chunk)):
                row = {key: float(value[i]) for key, value in per_frame.items()}
                if not bool(has_other[i]):
                    row['L1_other'] = row['gap'] = float('nan')
                rows.append(row)
        return rows


def summarize(rows):
    result = {'frames': len(rows)}
    for key in METRICS:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[key] = float(values.mean()) if len(values) else None
        result[f'{key}_se'] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
    return result


def task_mean(per_task):
    means = {}
    for key in METRICS:
        values = [entry[key] for entry in per_task.values() if entry.get(key) is not None]
        means[key] = float(np.mean(values)) if values else None
    return means


def run(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    checkpoint = load_checkpoint(args.checkpoint)
    adapter, train_config = checkpoint['adapter_config'], checkpoint.get('train_config', {})
    root = Path(args.dataset_path or checkpoint.get('conditioning', {}).get('dataset_root')
                or train_config['dataset_path']).resolve()
    frame_cache = args.frame_cache or train_config.get('frame_cache')
    heldout_root = Path(args.heldout_dataset).resolve() if args.heldout_dataset else root
    separate_root = heldout_root != root
    if args.split:
        split_path = Path(args.split)
    elif (heldout_root / 'isg_meta/eval_split.json').exists():
        split_path = heldout_root / 'isg_meta/eval_split.json'
    elif separate_root:
        raise SystemExit(f'--heldout-dataset {heldout_root} has no isg_meta/eval_split.json; pass --split')
    else:
        split_path = root / 'isg_meta/train_split.json'
    held_out, split_name = held_out_lists(split_path)
    tasks = list(checkpoint['task_map'].values())
    if adapter.get('task_conditioning', 'onehot') == 'onehot' and separate_root:
        raise SystemExit('One-hot checkpoints index tasks by the training dataset\'s task order; --heldout-dataset '
                         'supports checkpoints without the task one-hot only')
    settle = (checkpoint.get('settle_steps') or {}).get('spec', 'all')
    train_episodes = (checkpoint.get('episode_split') or {}).get('episodes')
    if not train_episodes:
        raise SystemExit('Checkpoint has no episode split; Tier A needs the training episodes')
    probe = Probe(checkpoint, args.device, args.batch_size, args.goal_mismatch_views)
    goal = probe.goal

    def dataset_for(task_names, episodes, subset_root=root, cache=frame_cache):
        dataset = B1KDataset(subset_root, task_names, probe.config['num_queries'], adapter['image_size'],
                             frame_cache=Path(cache).resolve() if cache else None,
                             goal_views=goal['goal_views'], goal_source=adapter.get('goal_source', 'episode_last'),
                             task_onehot=adapter.get('task_conditioning', 'onehot') == 'onehot', episodes=episodes,
                             settle_steps=settle, gripper_state=adapter.get('gripper_state', 'fingers'),
                             goal_mismatch_views=probe.mismatch_views)
        dataset.stats = checkpoint['normalization']
        return dataset

    result = {'format': FORMAT, 'checkpoint': str(Path(args.checkpoint).resolve()), 'step': int(checkpoint['step']),
              'run': (checkpoint.get('wandb') or {}).get('name'),
              'source_commit': checkpoint.get('conditioning', {}).get('source_commit'),
              'goal': {key: goal[key] for key in ('goal_fusion', 'goal_encoder', 'goal_tag_init', 'goal_stem_init',
                                                  'goal_tokens', 'lr_goal')},
              'probe': {'frames_per_task': args.frames_per_task, 'seed': args.seed,
                        'split': str(split_path), 'split_name': split_name,
                        'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
                        'heldout_dataset': str(heldout_root), 'settle_steps': settle, 'device': str(args.device),
                        'probe_commit': None}}
    if probe.mismatch_views:
        result['goal']['goal_mismatch_views'] = probe.mismatch_views
        result['probe']['goal_mismatch'] = ('mismatched views show another episode of the same task and subset in the '
                                            'own- and other-goal conditions, drawn per frame with the seed'
                                            + ('; evaluation only (the checkpoint trained on its own goals)'
                                               if probe.mismatch_evaluation_only else ''))
    for subset in ('heldout', 'train'):
        if subset == 'heldout':
            lists = {task: held_out[task] for task in tasks if task in held_out}
        else:
            chosen = set(int(e) for e in train_episodes)
            dataset = dataset_for(tasks, sorted(chosen))
            lists = {}
            for ep in dataset.episodes:
                lists.setdefault(checkpoint['task_map'][int(ep['task_index'])], []).append(int(ep['episode_index']))
            dataset.close()
        if not lists:
            result[subset] = {}
            continue
        separate = subset == 'heldout' and separate_root
        dataset = dataset_for(sorted(lists), sorted(e for episodes in lists.values() for e in episodes),
                              *((heldout_root, None) if separate else ()))
        if separate and dataset.settle:
            result['probe']['heldout_settle'] = {key: dataset.settle[key] for key in ('file', 'sha256', 'frames',
                                                                                      'recorded_frames')}
        per_task = {}
        for index, task in enumerate(tasks):
            if task not in lists:
                continue
            key = [args.seed, index, 0 if subset == 'heldout' else 1]
            rng = np.random.default_rng(np.random.SeedSequence(key))
            frames = stratified_frames(dataset, sorted(lists[task]), args.frames_per_task, rng)
            others = other_episodes(frames, sorted(lists[task]), rng)
            partners = (other_episodes(frames, sorted(lists[task]),
                                       np.random.default_rng(np.random.SeedSequence([*key, GOAL_MISMATCH_STREAM])))
                        if probe.mismatch_views else None)
            per_task[task] = dict(summarize(probe.evaluate(dataset, frames, others, partners)), episodes=len(lists[task]))
        dataset.close()
        result[subset] = {'tasks': per_task, 'mean': task_mean(per_task)}
    result['probe']['seconds'] = round(time.monotonic() - started, 1)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('--dataset-path', help='Default: the checkpoint\'s dataset root')
    p.add_argument('--frame-cache', help='Default: the checkpoint\'s training frame cache')
    p.add_argument('--heldout-dataset', help='Separate LeRobot root of the held-out episodes (tasks matched by name; '
                   'decoded from video). Default: the checkpoint\'s dataset root')
    p.add_argument('--split', help='Default: <held-out root>/isg_meta/eval_split.json if present, else (without '
                   '--heldout-dataset) <root>/isg_meta/train_split.json')
    p.add_argument('--goal-mismatch-views', nargs='+', choices=CAMERAS, metavar='CAMERA',
                   help='Evaluation only: these goal views show another episode of the same task and subset (the other '
                   'goal views keep their own), for a checkpoint trained on its own goals; write the result to a '
                   'separate --output so the standard Tier A results stay untouched')
    p.add_argument('--frames-per-task', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--output', help='Write the JSON result here (atomically) as well as to stdout')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    result = run(args)
    text = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.tmp')
        temporary.write_text(text)
        temporary.replace(output)
    print(text)


if __name__ == '__main__':
    main()
