#!/usr/bin/env python3
"""Decision-transfer study (plan decision log, 2026-10-08): do the relocalization architecture decisions hold on mental
rotation and object scaling? The pre-registered reading of held-out Tier A results at the final step (100k).

Starting from the selected option and undoing one decision at a time (waves/xfer1.json, machine 2, two seeds each):
  A0 early-wrist   early fusion, copy:0.5 stem, goal LR 1e-3, head+wrist goal views (the selected option)
  A1 early-copy05  A0 without the wrist goal views
  A2 early-zero    A1 with a zero-initialised goal stem
  A3 late-tagzero  late fusion (zero tag, goal LR 1e-4), head goal only
Relocalization's own values come from the reloc1 runs of the same recipe (r1-<config>-st0-s<seed>).

For each decision and task (`classify`), the paired difference chosen − alternative of held-out L1_own at both seeds
against the task's noise floor: relocalization's pre-registered delta = 0.0066; for mental rotation and object scaling
the largest absolute seed-0/seed-1 difference among the four configurations (two seeds give no better estimate; the
largest guards against a lucky small one). A decision transfers to a task when its class there equals its class on
relocalization.
"""

import json
from pathlib import Path

STEP = 100000
RELOC_DELTA = 0.0066
CONFIGS = ('early-wrist', 'early-copy05', 'early-zero', 'late-tagzero')
LABELS = {'early-wrist': 'A0 head+wrist (selected)', 'early-copy05': 'A1 head-only', 'early-zero': 'A2 zero init',
          'late-tagzero': 'A3 late fusion'}
TASKS = {'relocalization': ('camera_relocalization-standard', 'r1-{config}-st0-s{seed}'),
         'mental rotation': ('mental_rotation-banana', 'x1-mr-{config}-st0-s{seed}'),
         'object scaling': ('object_scaling-canned_food', 'x1-os-{config}-st0-s{seed}')}
DECISIONS = (('wrist goal views (A0 vs A1)', 'early-wrist', 'early-copy05'),
             ('copy:0.5 vs zero stem init (A1 vs A2)', 'early-copy05', 'early-zero'),
             ('early vs late fusion (A1 vs A3)', 'early-copy05', 'late-tagzero'))
SEEDS = (0, 1)


def result(runs_root, run, step):
    path = Path(runs_root) / run / 'tierA-eval50' / f'step_{step:08d}.json'
    return json.loads(path.read_text()) if path.exists() else None


def metric(record, task, key):
    value = ((record or {}).get('heldout', {}).get('tasks', {}).get(task) or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def latest(runs_root, run):
    steps = sorted(int(p.stem[5:]) for p in (Path(runs_root) / run / 'tierA-eval50').glob('step_*.json'))
    return steps[-1] if steps else None


def cell(runs_root, run, task, key):
    """The final-step value, else the latest step's in brackets, else a dash."""
    value = metric(result(runs_root, run, STEP), task, key)
    if value is not None:
        return f'{value:.4f}'
    step = latest(runs_root, run)
    value = metric(result(runs_root, run, step), task, key) if step is not None else None
    return '—' if value is None else f'[{step // 1000}k: {value:.4f}]'


def classify(diffs, delta):
    """Class of {seed: chosen − alternative held-out L1_own} (negative: the chosen option is better) against delta."""
    if all(d < -delta for d in diffs.values()):
        return 'chosen better'
    if all(d > delta for d in diffs.values()):
        return 'alternative better'
    if all(abs(d) <= delta for d in diffs.values()):
        return 'no difference'
    return 'mixed'


def render(runs_root):
    """(markdown section, final plan-log text or None while a final result is missing)."""
    lines = ['## Decision transfer: mental rotation and object scaling (machine 2, from 2026-10-08)', '',
             'Do the relocalization architecture decisions hold on other tasks? The single-task reloc1 recipe (100k steps) '
             'on mental_rotation-banana and object_scaling-canned_food, starting from the selected option and undoing one '
             'decision at a time: ' + '; '.join(f'{LABELS[c]} = `{c}`' for c in CONFIGS) + '. Held-out (eval50 test set) '
             'L1_own / goal gap at 100k, latest Tier A step in brackets while a run trains. Pre-registered reading in the '
             'plan\'s decision log (2026-10-08). Offline only.', '',
             '| Task | Seed | ' + ' | '.join(f'{LABELS[c]} L1_own' for c in CONFIGS) + ' | ' +
             ' | '.join(f'{c} gap' for c in CONFIGS) + ' |', '| --- | --- |' + ' ---: |' * 8]
    values, complete = {}, True
    for short, (task, pattern) in TASKS.items():
        for seed in SEEDS:
            runs = {c: pattern.format(config=c, seed=seed) for c in CONFIGS}
            cells = [cell(runs_root, runs[c], task, key) for key in ('L1_own', 'gap') for c in CONFIGS]
            lines.append(f'| {short} | s{seed} | ' + ' | '.join(cells) + ' |')
            for c in CONFIGS:
                value = metric(result(runs_root, runs[c], STEP), task, 'L1_own')
                values[(short, seed, c)] = value
                complete &= value is not None
    lines.append('')
    if not complete:
        lines.append('Reading: pending (every final result is needed).')
        return '\n'.join(lines), None
    deltas = {short: RELOC_DELTA if short == 'relocalization' else
              max(abs(values[(short, 0, c)] - values[(short, 1, c)]) for c in CONFIGS) for short in TASKS}
    rows, summary = ['| Decision | ' + ' | '.join(f'{t} (delta {deltas[t]:.4f})' for t in TASKS) + ' | Transfers to |',
                     '| --- |' + ' --- |' * (len(TASKS) + 1)], []
    for name, chosen, alternative in DECISIONS:
        classes = {}
        for short in TASKS:
            diffs = {s: values[(short, s, chosen)] - values[(short, s, alternative)] for s in SEEDS}
            classes[short] = (classify(diffs, deltas[short]), diffs)
        reference = classes['relocalization'][0]
        transfers = [t for t in TASKS if t != 'relocalization' and classes[t][0] == reference]
        rows.append(f'| {name} | ' + ' | '.join(f'{cls} ({", ".join(f"{d:+.4f}" for d in diffs.values())})'
                                                for cls, diffs in classes.values()) + f' | {", ".join(transfers) or "neither"} |')
        summary.append(f'{name}: relocalization {reference}; ' + '; '.join(f'{t} {classes[t][0]}' for t in TASKS if t != 'relocalization')
                       + f' (transfers to {", ".join(transfers) or "neither"})')
    lines += ['**Reading** (chosen − alternative held-out L1_own at s0, s1; negative: the chosen option is better):', '', *rows, '']
    return '\n'.join(lines), 'Decision transfer, final reading: ' + '. '.join(summary) + '.'
