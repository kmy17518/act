#!/usr/bin/env python3
"""Stop study, Method B: the completion verifier pipeline (b1k_stop_verifier.py) on the ACT frame cache.

  extract  frozen DINOv2 features of every `--stride`-th head-camera frame (and the last recorded frame, the goal) of
           every episode of the task groups, the recorded settle windows included, with the ACT val split of the
           policy runs (--episode-split auto, --val-fraction 0.1, --seed 0) -> <out>/features.npy, index.npz, episodes.json
  train    the verifier MLP on the training episodes' frames (positives: frame >= T_end) -> <out>/verifier.pt
  eval     sequential DONE rule on the validation episodes at 30 / --query-stride Hz: p_done >= tau for W consecutive
           queries; the (tau, W) grid, the fair setting (<= 2 % false stops, smallest median |t_done - T_end|),
           frame-level AUROC -> <out>/summary.json, traces.npz, episodes.jsonl (RESULT: line)
  combine  A + B from the traces of scripts/b1k/eval_stop_offline.py (A) and `eval` (B): A's gate proposes, B must
           confirm within --confirm-s seconds, else execution continues -> <out>/summary.json (RESULT: line)

    python scripts/b1k/stop_verifier.py extract --dataset-path ROOT --frame-cache CACHE --task-names camera_relocalization \\
        object_scaling mental_rotation cm iors_small --output /tmp/dev/runs/side/verifier/features
    python scripts/b1k/stop_verifier.py train --features .../features --output /tmp/dev/runs/side/verifier/verifier.pt
    python scripts/b1k/stop_verifier.py eval --features .../features --verifier .../verifier.pt --output .../eval
    python scripts/b1k/stop_verifier.py combine --a <eval_stop_offline output> --b .../eval --output .../combined
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from b1k_dataset import B1KDataset, CAMERAS, VIDEO_KEYS, expand_task_groups, resolve_episode_split  # noqa: E402
from b1k_stop_eval import (FPS, auroc, choose_setting, first_run_end, gate_proposals, method_metrics,  # noqa: E402
                           summary_line)
from b1k_stop_verifier import DEFAULT_BACKBONE, CompletionVerifier, FrameEncoder, history_rows  # noqa: E402

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
CONSECUTIVE = (1, 2, 3)


def sampled_frames(length, stride):
    return np.array(sorted(set(range(0, length, stride)) | {length - 1}), dtype=np.int64)


def load_features(directory, device):
    directory = Path(directory)
    features = torch.from_numpy(np.load(directory / 'features.npy', mmap_mode='r')[:]).to(device)
    index = np.load(directory / 'index.npz')
    episodes = json.loads((directory / 'episodes.json').read_text())
    return features, index['episode'], index['frame'], episodes


class SampleTable:
    """Row indices of the verifier inputs for every sampled frame of the chosen episodes."""
    def __init__(self, episode_column, frame_column, episodes, split, history, gap):
        current, past, goal, label, episode, frame = [], [], [], [], [], []
        rows_by_episode = {}
        for row, (ep, fr) in enumerate(zip(episode_column.tolist(), frame_column.tolist())):
            rows_by_episode.setdefault(ep, {})[fr] = row
        self.per_episode = {}
        for ep, entry in episodes['episodes'].items():
            if split and entry['split'] != split:
                continue
            row_of_frame = rows_by_episode[int(ep)]
            frames = np.array(sorted(row_of_frame), dtype=np.int64)
            goal_row = row_of_frame[int(frames[-1])]
            rows = []
            for fr in frames.tolist():
                rows.append(len(current))
                current.append(row_of_frame[fr])
                past.append(history_rows(fr, row_of_frame, history, gap, frames))
                goal.append(goal_row)
                label.append(fr >= entry['program_length'])
                episode.append(int(ep))
                frame.append(fr)
            self.per_episode[int(ep)] = (frames, np.array(rows))
        self.current, self.goal = np.array(current), np.array(goal)
        self.past = np.array(past).reshape(len(current), history)
        self.label = np.array(label, dtype=bool)
        self.episode, self.frame = np.array(episode), np.array(frame)

    def __len__(self):
        return len(self.current)

    def batch(self, features, indices):
        return (features[self.current[indices]], features[torch.as_tensor(self.past[indices], device=features.device)],
                features[self.goal[indices]])


def scores(model, features, table, batch_size=4096):
    out = np.empty(len(table), dtype=np.float32)
    with torch.inference_mode():
        for begin in range(0, len(table), batch_size):
            indices = np.arange(begin, min(begin + batch_size, len(table)))
            out[begin:begin + len(indices)] = model.p_done(*table.batch(features, indices)).float().cpu().numpy()
    return out


def extract(args):
    started = time.monotonic()
    root = args.dataset_path.resolve()
    task_names = expand_task_groups(root, args.task_names)
    episode_split = resolve_episode_split(root, args.episode_split, task_names)
    task_names = episode_split['tasks'] if episode_split else task_names
    from b1k_training import resolve_val_split
    val_split = resolve_val_split(root, task_names, episode_split, args.val_fraction, args.seed)
    val_episodes = set(val_split['val_episodes']) if val_split else set()
    selected = (val_split['train_episodes'] + val_split['val_episodes'] if val_split
                else (episode_split['episodes'] if episode_split else None))
    if args.episodes and selected is not None:
        selected = sorted(selected)[:args.episodes]
    dataset = B1KDataset(root, task_names, 1, tuple(args.image_size), frame_cache=args.frame_cache.resolve(),
                         episodes=selected, settle_steps='all', stop_dims=True, task_onehot=False)
    dataset._ensure_table()
    camera = CAMERAS.index(args.camera)
    key = VIDEO_KEYS[camera]
    plan = [(position, sampled_frames(int(dataset.lengths[position]), args.stride)) for position in range(len(dataset.episodes))]
    total = sum(len(frames) for _, frames in plan)
    encoder = FrameEncoder(args.backbone, args.device)
    features = np.empty((total, encoder.dim), dtype=np.float16)
    episode_column = np.empty(total, dtype=np.int64)
    frame_column = np.empty(total, dtype=np.int64)
    buffer = np.empty((args.batch_size, *dataset.image_size, 3), dtype=np.uint8)
    pending, written = [], 0  # (row, episode position, cache position)

    def flush():
        nonlocal written
        if not pending:
            return
        for slot, (_, position, cache_position) in enumerate(pending):
            ep = dataset.episodes[position]
            dataset.frame_cache.read(dataset.video_path(ep, key), cache_position, out=buffer[slot])
        encoded = encoder(buffer[:len(pending)]).cpu().numpy()
        for slot, (row, _, _) in enumerate(pending):
            features[row] = encoded[slot]
        written += len(pending)
        pending.clear()

    row = 0
    for position, frames in plan:
        ep = dataset.episodes[position]
        rows = int(dataset.starts[position]) + frames
        for frame, table_row in zip(frames.tolist(), rows.tolist()):
            episode_column[row], frame_column[row] = int(ep['episode_index']), frame
            pending.append((row, position, int(dataset._frame_rows[table_row, camera])))
            row += 1
            if len(pending) == args.batch_size:
                flush()
                if written % (50 * args.batch_size) == 0:
                    print(f'[{time.monotonic() - started:6.0f}s] {written}/{total} frames', flush=True)
    flush()
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'features.npy', features)
    np.savez(args.output / 'index.npz', episode=episode_column, frame=frame_column)
    episodes = {str(ep['episode_index']): {'task': dataset.task_map[int(ep['task_index'])], 'length': int(dataset.lengths[position]),
                                            'program_length': int(dataset.program_lengths[position]),
                                            'split': 'val' if int(ep['episode_index']) in val_episodes else 'train'}
                for position, ep in enumerate(dataset.episodes)}
    meta = {'backbone': args.backbone, 'dim': encoder.dim, 'stride': args.stride, 'camera': args.camera,
            'image_size': list(dataset.image_size), 'dataset': str(root), 'frame_cache': str(args.frame_cache),
            'task_names': task_names, 'val_fraction': args.val_fraction, 'seed': args.seed, 'fps': FPS,
            'frames': total, 'episodes': len(episodes), 'val_episodes': len(val_episodes)}
    (args.output / 'episodes.json').write_text(json.dumps({'meta': meta, 'episodes': episodes}))
    dataset.close()
    print(f'RESULT: EXTRACTED {total} frames of {len(episodes)} episodes ({len(val_episodes)} val) with {args.backbone} '
          f'({encoder.dim}-d) in {time.monotonic() - started:.0f}s -> {args.output}')


def train(args):
    started = time.monotonic()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    features, episode_column, frame_column, episodes = load_features(args.features, device)
    meta = episodes['meta']
    train_table = SampleTable(episode_column, frame_column, episodes, 'train', args.history, args.gap)
    val_table = SampleTable(episode_column, frame_column, episodes, 'val', args.history, args.gap)
    model = CompletionVerifier(meta['dim'], args.history, args.gap, args.hidden, args.dropout, meta['backbone']).to(device)
    positives = int(train_table.label.sum())
    pos_weight = torch.tensor((len(train_table) - positives) / max(1, positives), device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_table) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch)
    labels = torch.as_tensor(train_table.label, device=device, dtype=torch.float32)
    history = []
    print(f'training on {len(train_table)} frames ({positives} positives, pos_weight {float(pos_weight):.2f}) of '
          f'{len(train_table.per_episode)} episodes; validating on {len(val_table)} frames of {len(val_table.per_episode)}')
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_table), device=device)
        total = 0.0
        for step in range(steps_per_epoch):
            indices = order[step * args.batch_size:(step + 1) * args.batch_size].cpu().numpy()
            logits = model(*train_table.batch(features, indices))
            loss = criterion(logits, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss)
        model.eval()
        val_scores = scores(model, features, val_table) if len(val_table) else np.zeros(0)
        record = {'epoch': epoch + 1, 'loss': total / steps_per_epoch,
                  'val_auroc': auroc(val_scores, val_table.label) if len(val_table) else None,
                  'val_acc': float(((val_scores >= 0.5) == val_table.label).mean()) if len(val_table) else None}
        history.append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output, {'features': str(Path(args.features).resolve()), 'meta': meta, 'history': history,
                             'train_frames': len(train_table), 'train_positives': positives})
    last = history[-1]
    print(f"RESULT: TRAINED verifier ({args.epochs} epochs, {time.monotonic() - started:.0f}s): frame-level val AUROC "
          f"{last['val_auroc']}, acc {last['val_acc']} -> {args.output}")


def evaluate(args):
    started = time.monotonic()
    device = torch.device(args.device)
    features, episode_column, frame_column, episodes = load_features(args.features, device)
    model, saved = CompletionVerifier.load(args.verifier, device)
    table = SampleTable(episode_column, frame_column, episodes, args.split, model.history, model.gap)
    p = scores(model, features, table)
    stride = episodes['meta']['stride']
    if args.query_stride % stride:
        raise SystemExit(f'--query-stride {args.query_stride} must be a multiple of the extraction stride {stride}')
    traces, ids = {}, sorted(table.per_episode)
    for ep in ids:
        frames, rows = table.per_episode[ep]
        queries = frames % args.query_stride == 0
        traces[ep] = {'frames': frames[queries], 'p': p[rows][queries]}
    program = [episodes['episodes'][str(ep)]['program_length'] for ep in ids]
    length = [episodes['episodes'][str(ep)]['length'] for ep in ids]
    grid = []
    for tau in args.taus:
        for consecutive in args.consecutive:
            done = []
            for ep in ids:
                index = first_run_end(traces[ep]['p'] >= tau, consecutive)
                done.append(None if index is None else int(traces[ep]['frames'][index]))
            grid.append({'tau': tau, 'consecutive': consecutive, **method_metrics(done, program, length, args.after_s)})
    chosen = choose_setting(grid)
    frame_labels = table.frame >= np.array([episodes['episodes'][str(ep)]['program_length'] for ep in table.episode])
    summary = {'verifier': str(args.verifier), 'features': str(args.features), 'split': args.split, 'episodes': len(ids),
               'query_hz': FPS / args.query_stride, 'frame_auroc': auroc(p, frame_labels),
               'frame_acc': float(((p >= 0.5) == frame_labels).mean()),
               'methods': {'B': chosen}, 'B_grid': grid, 'verifier_config': model.config(),
               'tail_frames_median': float(np.median(np.subtract(length, program)))}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    with (args.output / 'episodes.jsonl').open('w') as handle:
        for ep, program_length, recorded in zip(ids, program, length):
            index = first_run_end(traces[ep]['p'] >= chosen['tau'], chosen['consecutive'])
            handle.write(json.dumps({'episode': ep, 'task': episodes['episodes'][str(ep)]['task'], 'length': recorded,
                                     'program_length': program_length,
                                     'B_done_step': None if index is None else int(traces[ep]['frames'][index])}) + '\n')
    np.savez_compressed(args.output / 'traces.npz', episodes=np.asarray(ids),
                        **{f'{key}_{ep}': traces[ep][key] for ep in ids for key in ('frames', 'p')})
    print(f"RESULT: {len(ids)} {args.split} episodes at {FPS / args.query_stride:.0f} Hz: frame AUROC {summary['frame_auroc']:.3f}; "
          + summary_line(f"B tau {chosen['tau']} W {chosen['consecutive']}", chosen)
          + f' ({time.monotonic() - started:.0f}s) -> {args.output}')


def combine(args):
    a_summary = json.loads((args.a / 'summary.json').read_text())
    b_summary = json.loads((args.b / 'summary.json').read_text())
    a_rows = {row['episode']: row for row in map(json.loads, (args.a / 'episodes.jsonl').read_text().splitlines())}
    a_traces, b_traces = np.load(args.a / 'traces.npz'), np.load(args.b / 'traces.npz')
    serving, b = a_summary['serving'], b_summary['methods']['B']
    tau, consecutive = (args.tau if args.tau is not None else b['tau']), (args.consecutive or b['consecutive'])
    confirm = int(round(args.confirm_s * FPS))
    ids = sorted(set(a_traces['episodes'].tolist()) & set(b_traces['episodes'].tolist()))
    if not ids:
        raise SystemExit('No common episodes between the A and B traces')
    program = [a_rows[ep]['program_length'] for ep in ids]
    length = [a_rows[ep]['length'] for ep in ids]
    done = {'A': [], 'B': [], 'A+B': []}
    for ep in ids:
        proposals = gate_proposals(a_traces[f'stop_{ep}'], a_traces[f'progress_{ep}'], serving['stop_threshold'],
                                   serving['progress_threshold'], serving['hold_steps'])
        frames, p = b_traces[f'frames_{ep}'], b_traces[f'p_{ep}']
        confirmations = np.zeros(len(frames), dtype=bool)
        run = 0
        for index, flag in enumerate(p >= tau):
            run = run + 1 if flag else 0
            confirmations[index] = run >= consecutive
        done['A'].append(int(np.argmax(proposals)) if proposals.any() else None)
        done['B'].append(int(frames[np.argmax(confirmations)]) if confirmations.any() else None)
        both = None
        for frame, confirmed in zip(frames.tolist(), confirmations):
            if confirmed and proposals[max(0, frame - confirm):frame + 1].any():
                both = frame
                break
        done['A+B'].append(both)
    summary = {'a': str(args.a), 'b': str(args.b), 'episodes': len(ids), 'confirm_s': args.confirm_s,
               'b_setting': {'tau': tau, 'consecutive': consecutive}, 'a_gate': serving,
               'methods': {name: method_metrics(steps, program, length, args.after_s) for name, steps in done.items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=1))
    with (args.output / 'episodes.jsonl').open('w') as handle:
        for index, ep in enumerate(ids):
            handle.write(json.dumps({'episode': ep, 'program_length': program[index], 'length': length[index],
                                     **{f'{name}_done_step': steps[index] for name, steps in done.items()}}) + '\n')
    print(f'RESULT: {len(ids)} episodes: ' + '; '.join(summary_line(name, summary['methods'][name]) for name in ('A', 'B', 'A+B'))
          + f' -> {args.output}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--dataset-path', type=Path, required=True)
    e.add_argument('--frame-cache', type=Path, required=True)
    e.add_argument('--task-names', nargs='+', required=True)
    e.add_argument('--episode-split', default='auto')
    e.add_argument('--val-fraction', type=float, default=0.1)
    e.add_argument('--seed', type=int, default=0)
    e.add_argument('--image-size', type=int, nargs=2, default=[224, 224], help='The frame cache size')
    e.add_argument('--camera', default='zed_link', choices=CAMERAS)
    e.add_argument('--stride', type=int, default=5, help='Every STRIDE-th frame (default 5: 6 Hz), plus the last frame')
    e.add_argument('--backbone', default=DEFAULT_BACKBONE)
    e.add_argument('--batch-size', type=int, default=256)
    e.add_argument('--device', default='cuda')
    e.add_argument('--episodes', type=int, help='The first N selected episodes only (smoke runs)')
    e.add_argument('--output', type=Path, required=True)
    t = sub.add_parser('train')
    t.add_argument('--features', type=Path, required=True)
    t.add_argument('--output', type=Path, required=True, help='verifier.pt')
    t.add_argument('--history', type=int, default=3)
    t.add_argument('--gap', type=int, default=30, help='Frames between history frames (30 = 1 s)')
    t.add_argument('--hidden', type=int, default=512)
    t.add_argument('--dropout', type=float, default=0.1)
    t.add_argument('--epochs', type=int, default=8)
    t.add_argument('--batch-size', type=int, default=2048)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--weight-decay', type=float, default=1e-2)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--device', default='cuda')
    v = sub.add_parser('eval')
    v.add_argument('--features', type=Path, required=True)
    v.add_argument('--verifier', type=Path, required=True)
    v.add_argument('--split', default='val', choices=['val', 'train'])
    v.add_argument('--query-stride', type=int, default=10, help='Frames between verifier queries (10 = 3 Hz)')
    v.add_argument('--taus', type=float, nargs='+', default=list(TAUS))
    v.add_argument('--consecutive', type=int, nargs='+', default=list(CONSECUTIVE))
    v.add_argument('--after-s', type=float, default=5.0)
    v.add_argument('--device', default='cuda')
    v.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('combine')
    c.add_argument('--a', type=Path, required=True, help='Output directory of scripts/b1k/eval_stop_offline.py')
    c.add_argument('--b', type=Path, required=True, help='Output directory of `eval`')
    c.add_argument('--confirm-s', type=float, default=2.0)
    c.add_argument('--tau', type=float, help="B's threshold (default: the eval's fair setting)")
    c.add_argument('--consecutive', type=int, help="B's consecutive queries (default: the eval's fair setting)")
    c.add_argument('--after-s', type=float, default=5.0)
    c.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    {'extract': extract, 'train': train, 'eval': evaluate, 'combine': combine}[args.command](args)


if __name__ == '__main__':
    main()
