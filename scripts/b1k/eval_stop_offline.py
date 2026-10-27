#!/usr/bin/env python3
"""Offline stop evaluation of an ACT checkpoint on its held-out validation episodes (stop study, Method A + baseline).

Every validation episode is replayed frame by frame, every recorded frame included (the settle window after T_end),
through the policy server's own `Session` (b1k_server.py): the same chunking or temporal aggregation, the same
`StopGate` as at deployment, fed the recorded observations (frame-cache images at the training size, raw states, the
episode's goal image). T_end is the episode's program length (isg_meta/settle_windows.json). Per episode and method
the script records when DONE was declared and reports

  per-frame AUROC of the stop signal against [t >= T_end], MAE of the progress signal against clip(t / T_end, 0, 1),
  median |t_done - T_end| in seconds, false stops (DONE before T_end; also with a 0.5 s tolerance), missed stops
  (no DONE within the recorded frames after T_end, and within 5 s where the recording allows), stop latency.

Methods: A (a --stop-dims checkpoint's StopGate, thresholds and hold steps as served) and the baseline stationarity
heuristic on the executed absolute actions of any checkpoint (|commanded joint target - joint state| and |base
velocity| below eps for N consecutive steps; a grid of eps / N, the "fair" setting = the smallest median |t - T_end|
among those with at most 2 % false stops). Per-step traces (stop, progress, motion) are saved for combining with the
verifier (Method B) and for failure analysis.

    python scripts/b1k/eval_stop_offline.py --checkpoint RUN/export_queue/eval/step_00200000.pt \\
        --dataset-path /tmp/dev/datasets/isg-1020-b1k --frame-cache /tmp/dev/datasets/isg-1020-b1k-act-frame-cache-224x224 \\
        --temporal-agg --output /tmp/dev/runs/side/stop-eval/<run>
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from b1k_dataset import ACTION_DELTA, B1KDataset, GOAL_OBS_KEYS, OBS_KEYS, VIDEO_KEYS, extract_state  # noqa: E402
from b1k_server import DEFAULT_GRIPPER_SNAP, PolicyPredictor, Session, StopGate  # noqa: E402
from b1k_stop_eval import FPS, auroc, choose_setting, first_run_end, method_metrics, summary_line  # noqa: E402
from b1k_training import goal_config, load_checkpoint  # noqa: E402

BASELINE_EPS = (0.005, 0.01, 0.02, 0.05)  # rad (joint target - joint state), also m/s or rad/s on the base
BASELINE_STEPS = (15, 30, 60, 90)  # consecutive control steps


class EpisodeFrames:
    """Recorded observations of a dataset's episodes in the policy server's observation layout."""
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset.frame_cache is not None:
            dataset._ensure_table()

    def observation(self, position, frame):
        ds = self.dataset
        ep = ds.episodes[position]
        if ds.frame_cache is not None:
            row = int(ds.starts[position]) + frame
            state = np.asarray(ds._table['state'][row], dtype=np.float32)
            images = np.empty((len(VIDEO_KEYS), *ds.image_size, 3), dtype=np.uint8)
            for camera, key in enumerate(VIDEO_KEYS):
                ds.frame_cache.read(ds.video_path(ep, key), int(ds._frame_rows[row, camera]), out=images[camera])
        else:
            images, state, _, _, _ = ds.raw_sample(int(ep['episode_index']), frame)  # source resolution; Session resizes
        goals = ds.goal_for(position).numpy()
        observation = {'robot_r1::proprio': state, **{OBS_KEYS[c]: images[c] for c in range(len(VIDEO_KEYS))}}
        for view, image in zip(ds.goal_views, goals):
            observation[GOAL_OBS_KEYS[view]] = image
        return observation, int(ep['task_index'])


def batch_observations(frames, positions, step):
    """One batched observation of the slots' episodes at control step `step` (slots past their end repeat their last
    frame; their outputs are ignored)."""
    singles, tasks = [], []
    for position in positions:
        frame = min(step, int(frames.dataset.lengths[position]) - 1)
        observation, task = frames.observation(position, frame)
        singles.append(observation)
        tasks.append(task)
    batch = {key: np.stack([single[key] for single in singles]) for key in singles[0]}
    batch['task_id'] = np.asarray(tasks, dtype=np.int64)
    return batch


def motion_measure(action, state):
    """Per-slot baseline motion: the largest commanded joint displacement |target - state| (rad) and base speed."""
    proprio = extract_state(np.asarray(state, dtype=np.float32), 'sum')
    joints = np.abs(action[:, ACTION_DELTA['joint_action_dims']] - proprio[:, ACTION_DELTA['joint_state_dims']]).max(axis=1)
    base = np.abs(action[:, ACTION_DELTA['absolute_action_dims']]).max(axis=1)
    return np.maximum(joints, base)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset-path', type=Path, help="Default: the run's dataset path")
    p.add_argument('--frame-cache', type=Path, help="Default: the run's frame cache when it exists, else video decoding")
    p.add_argument('--device', default='cuda')
    p.add_argument('--episodes', type=int, help='Evaluate the first N validation episodes only (default: all)')
    p.add_argument('--slots', type=int, default=64, help='Episodes replayed at once (batch size)')
    p.add_argument('--temporal-agg', action='store_true', help='Serve with temporal aggregation (the protocol default)')
    p.add_argument('--action-horizon', type=int, help='Chunked serving: actions executed per chunk (default min(16, chunk))')
    p.add_argument('--stop-positions', type=int, help='Chunked serving: chunk steps averaged into the stop / progress signals')
    p.add_argument('--stop-threshold', type=float, default=0.5)
    p.add_argument('--progress-threshold', type=float, default=0.9)
    p.add_argument('--stop-hold-steps', type=int, default=30)
    p.add_argument('--gripper-snap-threshold', type=float, default=DEFAULT_GRIPPER_SNAP)
    p.add_argument('--after-s', type=float, default=5.0, help='Missed stop: no DONE this many seconds after T_end')
    p.add_argument('--output', type=Path, required=True, help='Directory: summary.json, episodes.jsonl, traces.npz')
    args = p.parse_args()

    started = time.monotonic()
    checkpoint = load_checkpoint(args.checkpoint)
    if not checkpoint.get('val_split'):
        raise SystemExit(f'{args.checkpoint} was trained without a validation split')
    config, adapter, train = checkpoint['model_config'], checkpoint['adapter_config'], checkpoint['train_config']
    root = (args.dataset_path or Path(train['dataset_path'])).resolve()
    cache = args.frame_cache or (Path(train['frame_cache']) if train.get('frame_cache') else None)
    cache = cache.resolve() if cache is not None and cache.exists() else None
    goal = goal_config(config)
    episodes = list(checkpoint['val_split']['val_episodes'])[:args.episodes]
    dataset = B1KDataset(root, list(checkpoint['task_map'].values()), config['num_queries'], adapter['image_size'],
                         timestamp_tolerance=adapter['timestamp_tolerance'], frame_cache=cache,
                         goal_views=goal['goal_views'], goal_source=adapter.get('goal_source', 'episode_last'),
                         task_onehot=adapter.get('task_conditioning', 'onehot') == 'onehot', episodes=episodes,
                         settle_steps='all', gripper_state=adapter.get('gripper_state', 'fingers'), stop_dims=True)
    if dataset.task_map != checkpoint['task_map']:
        raise SystemExit('The dataset task map differs from the checkpoint')
    frames = EpisodeFrames(dataset)
    predictor = PolicyPredictor(checkpoint, args.device)
    gate = StopGate(args.stop_threshold, args.progress_threshold, args.stop_hold_steps)
    session = Session(predictor, checkpoint, 1 if args.temporal_agg else args.action_horizon, None, args.temporal_agg,
                      None, args.gripper_snap_threshold, gate, args.stop_positions)
    has_signals = session.stop_signals
    order = sorted(range(len(dataset.episodes)), key=lambda position: int(dataset.lengths[position]))
    traces = {}  # episode -> dict of per-step arrays
    for begin in range(0, len(order), args.slots):
        positions = order[begin:begin + args.slots]
        lengths = [int(dataset.lengths[position]) for position in positions]
        longest = max(lengths)
        stop = np.full((len(positions), longest), np.nan, dtype=np.float32)
        progress = np.full((len(positions), longest), np.nan, dtype=np.float32)
        done = np.zeros((len(positions), longest), dtype=bool)
        motion = np.full((len(positions), longest), np.nan, dtype=np.float32)
        session.reset()
        for step in range(longest):
            batch = batch_observations(frames, positions, step)
            action = session.act(batch)
            motion[:, step] = motion_measure(action, batch['robot_r1::proprio'])
            if has_signals:
                stop[:, step] = session.signals['stop']
                progress[:, step] = session.signals['progress']
                done[:, step] = session.signals['done']
        for slot, position in enumerate(positions):
            ep = dataset.episodes[position]
            length = lengths[slot]
            traces[int(ep['episode_index'])] = {
                'task': dataset.task_map[int(ep['task_index'])], 'length': length,
                'program_length': int(dataset.program_lengths[position]),
                'stop': stop[slot, :length], 'progress': progress[slot, :length], 'done': done[slot, :length],
                'motion': motion[slot, :length]}
        print(f'[{time.monotonic() - started:6.0f}s] replayed {min(begin + args.slots, len(order))}/{len(order)} episodes',
              flush=True)

    ids = sorted(traces)
    program = [traces[i]['program_length'] for i in ids]
    length = [traces[i]['length'] for i in ids]
    summary = {'checkpoint': str(args.checkpoint), 'step': checkpoint.get('step'), 'episodes': len(ids),
               'serving': {'temporal_agg': args.temporal_agg, 'action_horizon': session.action_horizon,
                           'stop_positions': session.stop_positions, **gate.config()},
               'fps': FPS, 'tail_frames_median': float(np.median(np.subtract(length, program))),
               'methods': {}}
    rows = {i: {'episode': i, 'task': traces[i]['task'], 'length': traces[i]['length'],
                'program_length': traces[i]['program_length']} for i in ids}
    if has_signals:
        t = [np.arange(traces[i]['length']) for i in ids]
        labels = np.concatenate([steps >= traces[i]['program_length'] for i, steps in zip(ids, t)])
        stop_all = np.concatenate([traces[i]['stop'] for i in ids])
        progress_all = np.concatenate([traces[i]['progress'] for i in ids])
        target_all = np.concatenate([np.clip(steps / traces[i]['program_length'], 0, 1) for i, steps in zip(ids, t)])
        done_steps = [int(np.argmax(traces[i]['done'])) if traces[i]['done'].any() else None for i in ids]
        method_a = method_metrics(done_steps, program, length, args.after_s)
        method_a.update({'frame_auroc_stop': auroc(stop_all, labels),
                         'frame_mae_progress': float(np.abs(progress_all - target_all).mean()),
                         'frame_stop_acc': float(((stop_all >= 0.5) == labels).mean())})
        summary['methods']['A'] = method_a
        for i, t_done in zip(ids, done_steps):
            rows[i]['A_done_step'] = t_done
    grid = []
    for eps in BASELINE_EPS:
        for steps in BASELINE_STEPS:
            done_steps = [first_run_end(traces[i]['motion'] < eps, steps) for i in ids]
            entry = {'eps': eps, 'steps': steps, **method_metrics(done_steps, program, length, args.after_s)}
            grid.append(entry)
            for i, t_done in zip(ids, done_steps):
                rows[i][f'baseline_{eps}_{steps}_done_step'] = t_done
    chosen = choose_setting(grid)
    summary['methods']['baseline'] = chosen
    summary['baseline_grid'] = grid
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    with (args.output / 'episodes.jsonl').open('w') as handle:
        for i in ids:
            handle.write(json.dumps(rows[i]) + '\n')
    np.savez_compressed(args.output / 'traces.npz', episodes=np.asarray(ids),
                        **{f'{key}_{i}': traces[i][key] for i in ids for key in ('stop', 'progress', 'done', 'motion')})
    a = summary['methods'].get('A')
    parts = []
    if a:
        parts.append(summary_line('A', a) + f" (frame AUROC {a['frame_auroc_stop']:.3f}, progress MAE {a['frame_mae_progress']:.3f})")
    parts.append(summary_line(f"baseline eps {chosen['eps']} N {chosen['steps']}", chosen))
    print(f"RESULT: {len(ids)} val episodes, step {checkpoint.get('step')}: " + '; '.join(parts))
    print(f'wrote {args.output}/summary.json, episodes.jsonl, traces.npz in {time.monotonic() - started:.0f}s')


if __name__ == '__main__':
    main()
