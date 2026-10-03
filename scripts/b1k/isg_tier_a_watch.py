#!/usr/bin/env python3
"""Tier A watcher for an ISG wave manifest: probe every new eval-interval checkpoint and publish the metrics.

Every poll, for each run of the manifest (runs and deferred runs) it looks for full checkpoints
`<runs_root>/<run>/step_XXXXXXXX.pt` at multiples of the manifest's eval_every that have no
`<run>/tierA/step_XXXXXXXX.json` yet, runs scripts/b1k/isg_tier_a.py on the run's GPU (a separate process, so its
memory is returned after each checkpoint), adds the training-log context (mean training L1 over the last 500 steps,
s/step, peak GiB) and logs everything to the W&B run `<run>-ta` (name `<run>-tierA`, group `<run>`) at the
checkpoint's step. A second W&B run per training run avoids two writers on the trainer's own run.
`<logs_root>/tier-a-summary.json` collects every result for reporting.

    isg_tier_a_watch.py MANIFEST [--poll-seconds 120] [--once]
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CHECKOUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHECKOUT / 'scripts/b1k'))
from isg_wave import load, paths  # noqa: E402

SHORT = {'camera_relocalization-standard': 'reloc',
         'configuration_matching-articulation_open_large_scale-dishwasher': 'dishwasher',
         'configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table': 'blender',
         'alignment-axial-board_game-breakfast_table': 'boardgame',
         'camera_relocalization-obstructed-footstool_1': 'reloc_obstructed'}


def training_context(run_dir, step):
    """Mean training L1/KL over the 500 steps up to `step`, median s/step over them, and the peak reserved GiB."""
    records = []
    with (run_dir / 'metrics.jsonl').open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if step - 500 < record['step'] <= step:
                records.append(record)
    if not records:
        return {}
    times = sorted(r['timing/step_s'] for r in records[1:]) or [records[0]['timing/step_s']]
    return {'train_l1_last500': sum(r['l1'] for r in records) / len(records),
            'train_kl_last500': sum(r['kl'] for r in records) / len(records),
            's_per_step': times[len(times) // 2],
            'peak_gib': max(r.get('gpu/peak_reserved_bytes', 0) for r in records) / 2 ** 30}


def flatten(result):
    flat = {}
    for subset in ('heldout', 'train'):
        block = result.get(subset) or {}
        for key, value in (block.get('mean') or {}).items():
            if value is not None:
                flat[f'tierA/{subset}/{key}'] = value
        for task, entry in (block.get('tasks') or {}).items():
            for key, value in entry.items():
                if isinstance(value, (int, float)) and value is not None and not key.endswith('_se'):
                    flat[f'tierA/{subset}/{SHORT.get(task, task)}/{key}'] = value
    for key, value in (result.get('training') or {}).items():
        flat[f'tierA/training/{key}'] = value
    return flat


def log_wandb(manifest, name, result):
    if not os.environ.get('WANDB_API_KEY'):
        return 'skipped (no WANDB_API_KEY)'
    import wandb
    run = wandb.init(project=manifest['wandb']['project'], entity=manifest['wandb']['entity'], id=f'{name}-ta',
                     name=f'{name}-tierA', group=name, job_type='tier-a', resume='allow',
                     dir=str(Path(manifest['logs_root']) / 'wandb-tier-a'), reinit='finish_previous',
                     config={'training_run': name, 'probe': result['probe'], 'goal': result['goal']})
    try:
        run.log(flatten(result), step=int(result['step']))
    finally:
        run.finish()
    return f'{name}-ta'


def pending(manifest, name):
    run_dir = paths(manifest, name)['run']
    every = int(manifest['common']['eval_every'])
    found = []
    for path in sorted(run_dir.glob('step_*.pt')):
        match = re.fullmatch(r'step_(\d{8,})\.pt', path.name)
        if not match or path.is_symlink():
            continue
        step = int(match.group(1))
        if step % every == 0 and step > 0 and not (run_dir / 'tierA' / f'step_{step:08d}.json').exists():
            found.append((step, path))
    return found


def probe(manifest, name, run, step, checkpoint):
    run_dir = paths(manifest, name)['run']
    output = run_dir / 'tierA' / f'step_{step:08d}.json'
    raw = output.with_name(output.stem + '.probe.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(run.get('gpu', 0)), OMP_NUM_THREADS='4')
    started = time.time()
    subprocess.run(['taskset', '-c', '120-143', str(CHECKOUT / '.venv/bin/python'), '-u',
                    str(CHECKOUT / 'scripts/b1k/isg_tier_a.py'), str(checkpoint), '--output', str(raw),
                    '--device', 'cuda', '--batch-size', '64'],
                   check=True, env=env, stdout=subprocess.DEVNULL, cwd=CHECKOUT)
    result = json.loads(raw.read_text())
    result['training'] = training_context(run_dir, step)
    result['wandb_tier_a'] = log_wandb(manifest, name, result)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(output)
    raw.unlink(missing_ok=True)
    print(json.dumps({'event': 'tier_a', 'run': name, 'step': step, 'seconds': round(time.time() - started),
                      'heldout': result['heldout'].get('mean'), 'train': result['train'].get('mean')}), flush=True)


def write_summary(manifest):
    summary = {}
    for name in manifest['_runs']:
        directory = paths(manifest, name)['run'] / 'tierA'
        if directory.exists():
            summary[name] = {int(p.stem[5:]): json.loads(p.read_text())
                             for p in sorted(directory.glob('step_*.json')) if not p.stem.endswith('.probe')}
    target = Path(manifest['logs_root']) / 'tier-a-summary.json'
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(summary, indent=1, sort_keys=True))
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('manifest')
    parser.add_argument('--poll-seconds', type=float, default=120)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        manifest = load(args.manifest)  # re-read: runs may be added during the wave
        for name, run in manifest['_runs'].items():
            for step, checkpoint in pending(manifest, name):
                try:
                    probe(manifest, name, run, step, checkpoint)
                except Exception as exc:  # keep watching the other runs
                    print(json.dumps({'event': 'tier_a_failed', 'run': name, 'step': step,
                                      'error': f'{type(exc).__name__}: {exc}'[:500]}), flush=True)
        write_summary(manifest)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    sys.exit(main())
