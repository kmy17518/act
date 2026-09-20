#!/usr/bin/env python3
"""Summarize B1K training runs (ACT metrics.jsonl or Diffusion Policy train.jsonl) into a Markdown table.

    summarize_runs.py --window 100 LABEL=RUN_DIR [LABEL=RUN_DIR ...]

For every run: the final step, the mean loss terms over the last `--window` steps (ACT: l1/kl/loss; DP: loss),
the same mean over steps 1..window, the median step time, the peak allocated GPU memory, wall time, and the
recorded trainer commit / conditioning regime when the run directory or its neighbours carry them.
"""

import argparse
import json
from pathlib import Path
import statistics


def read_records(run_dir):
    for name in ('metrics.jsonl', 'train.jsonl'):
        path = run_dir / name
        if path.is_file():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            return name, rows
    raise FileNotFoundError(f'{run_dir}: no metrics.jsonl / train.jsonl')


def summarize(label, run_dir, window):
    run_dir = Path(run_dir)
    kind, rows = read_records(run_dir)
    last = [r for r in rows if r['step'] > rows[-1]['step'] - window]
    first = [r for r in rows if r['step'] <= window]
    if kind == 'metrics.jsonl':  # ACT
        loss_keys, step_key, peak_key = ('l1', 'kl', 'loss'), 'timing/step_s', 'gpu/peak_allocated_bytes'
    else:  # DP
        loss_keys, step_key, peak_key = ('loss',), 'step_s', 'gpu_peak_allocated_bytes'
    means = {key: statistics.mean(r[key] for r in last) for key in loss_keys}
    early = {key: statistics.mean(r[key] for r in first) for key in loss_keys}
    steady = rows[len(rows) // 10:] or rows
    step_s = statistics.median(r[step_key] for r in steady)
    peak = max(r.get(peak_key, 0) for r in rows) / 2 ** 30
    elapsed = rows[-1]['elapsed_s'] / 3600
    commit = None
    for candidate in (run_dir / 'trainer_commit.txt', run_dir.with_name(run_dir.name + '.trainer_commit.txt')):
        if candidate.is_file():
            commit = candidate.read_text().split()[0][:9]
    regime = None
    for candidate in (run_dir / 'run.json', run_dir / 'config.json'):
        if candidate.is_file():
            record = json.loads(candidate.read_text())
            regime = (record.get('conditioning') or {}).get('regime')
    return {'label': label, 'steps': rows[-1]['step'], 'final': means, 'initial': early, 'step_s': step_s,
            'peak_gib': peak, 'hours': elapsed, 'commit': commit, 'regime': regime, 'kind': kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', help='LABEL=RUN_DIR')
    parser.add_argument('--window', type=int, default=100)
    args = parser.parse_args()
    results = [summarize(*spec.split('=', 1), window=args.window) for spec in args.runs]
    act = [r for r in results if r['kind'] == 'metrics.jsonl']
    dp = [r for r in results if r['kind'] == 'train.jsonl']
    if act:
        print(f'| Run | commit | regime | steps | L1 (last {args.window}) | KL | loss | L1 (first {args.window}) | s/step | peak GiB | hours |')
        print('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
        for r in act:
            print(f"| {r['label']} | {r['commit']} | {r['regime']} | {r['steps']} | {r['final']['l1']:.4f} | {r['final']['kl']:.4f} | "
                  f"{r['final']['loss']:.4f} | {r['initial']['l1']:.4f} | {r['step_s']:.3f} | {r['peak_gib']:.0f} | {r['hours']:.2f} |")
    if dp:
        print(f'| Run | commit | regime | steps | loss (last {args.window}) | loss (first {args.window}) | s/step | peak GiB | hours |')
        print('| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
        for r in dp:
            print(f"| {r['label']} | {r['commit']} | {r['regime']} | {r['steps']} | {r['final']['loss']:.5f} | {r['initial']['loss']:.4f} | "
                  f"{r['step_s']:.3f} | {r['peak_gib']:.0f} | {r['hours']:.2f} |")


if __name__ == '__main__':
    main()
