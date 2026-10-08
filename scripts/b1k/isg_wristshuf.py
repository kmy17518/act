#!/usr/bin/env python3
"""Wrist-goal control (plan decision log, 2026-10-08): the pre-registered reading of the held-out Tier A results.

Is the wrist-goal gain from wrist goal *content* or from the extra training of the shared goal filters? The control
runs (`--goal-mismatch-views left_realsense_link right_realsense_link`) see their own head goal and the wrist goals of
another training episode of the same task, drawn per sample.

  single-task (reloc1, 100k):  control r1-early-wristshuf-st0-s<k>, wrist r1-early-wrist-st0-s<k>, head-only
                               r1-early-copy05-st0-s<k>
  four-task (wave 2, 50k):     control w2-early-wristshuf-lrg1e-3-s<k>, wrist w2-early-wrist-lrg1e-3-s<k>, head-only
                               w1-early-copy05-lrg1e-3-s<k>

Primary: held-out relocalization L1_own at the final step, paired by seed, against the relocalization noise floor
delta = 0.0066 (`reading`). The collector renders `render()` into docs/report.md every cycle and logs the final
reading in the plan once every final result exists.
"""

import json
from pathlib import Path

DELTA = 0.0066
RELOC = 'camera_relocalization-standard'
TASKS = {RELOC: 'relocalization', 'configuration_matching-articulation_open_large_scale-dishwasher': 'dishwasher',
         'configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table': 'blender',
         'alignment-axial-board_game-breakfast_table': 'board game (control task)'}
STUDIES = {
    'single-task': {'step': 100000, 'runs': {s: {'control': f'r1-early-wristshuf-st0-s{s}', 'wrist': f'r1-early-wrist-st0-s{s}',
                                                  'head-only': f'r1-early-copy05-st0-s{s}'} for s in (0, 1)}},
    'four-task': {'step': 50000, 'runs': {s: {'control': f'w2-early-wristshuf-lrg1e-3-s{s}', 'wrist': f'w2-early-wrist-lrg1e-3-s{s}',
                                               'head-only': f'w1-early-copy05-lrg1e-3-s{s}'} for s in (0, 1)}},
}
ROLES = ('head-only', 'wrist', 'control')


def tier_a(runs_root, run, step):
    path = Path(runs_root) / run / 'tierA-eval50' / f'step_{step:08d}.json'
    return json.loads(path.read_text()) if path.exists() else None


def latest(runs_root, run):
    steps = sorted(int(p.stem[5:]) for p in (Path(runs_root) / run / 'tierA-eval50').glob('step_*.json'))
    return steps[-1] if steps else None


def metric(result, task, key):
    value = ((result or {}).get('heldout', {}).get('tasks', {}).get(task) or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else None


def cell(runs_root, run, final, task, key):
    """A table cell: the final-step value, else the latest step's in brackets, else a dash."""
    value = metric(final, task, key)
    if value is not None:
        return f'{value:.4f}'
    step = latest(runs_root, run)
    value = metric(tier_a(runs_root, run, step), task, key) if step is not None else None
    return '—' if value is None else f'[{step // 1000}k: {value:.4f}]'


def reading(values, delta=DELTA):
    """Pre-registered single-task reading of {seed: (head-only, wrist, control)} held-out relocalization L1_own."""
    per_seed = '; '.join(f's{s}: control effect {h - c:+.4f} / gain {h - w:+.4f}' for s, (h, w, c) in sorted(values.items()))
    noisy = [s for s, (h, _, c) in sorted(values.items()) if c - h > delta]
    if noisy:
        return 'inconclusive', (f'the control is worse than head-only by more than delta at seed(s) {noisy}: the random wrist '
                                f'goals act as input noise and the test is inconclusive (follow-up: a fixed per-episode '
                                f'mismatch). {per_seed}')
    if all(abs(c - w) <= delta and h - c > delta for h, w, c in values.values()):
        return 'not wrist content', (f'the control is within delta of wrist and below head-only by more than delta at both '
                                     f'seeds: the gain does not come from wrist goal content (shared-filter training or '
                                     f'another non-content effect). {per_seed}')
    if all(abs(c - h) <= delta and c - w > delta for h, w, c in values.values()):
        return 'wrist content', (f'the control is within delta of head-only and wrist beats it by more than delta at both '
                                 f'seeds: wrist goal content drives the gain. {per_seed}')
    return 'mixed', f'neither pattern holds at both seeds; control effect / gain per seed: {per_seed}'


def render(runs_root):
    """(markdown section, final plan-log text or None while a final result is missing)."""
    lines = ['## Wrist-goal control (machine 1, from 2026-10-08)', '',
             'Question: is the wrist-goal gain on held-out relocalization from wrist goal *content* or from the extra '
             'training of the shared goal filters (with wrist goals the goal half of the paired stem trains on three '
             'camera-goal pairs per sample instead of one)? Control: the head camera keeps its own goal, both wrist '
             'cameras show the last frames of another training episode of the same task, drawn per sample '
             '(`--goal-mismatch-views`); frames, batches, initialization and parameter count are those of the wrist run. '
             'Tier A applies the same rule on held-out frames, so the control\'s gap measures head-goal reliance. '
             f'Pre-registered reading in the plan\'s decision log (2026-10-08); delta = {DELTA}. Offline only (no lab).', '']
    final = {}
    for study, spec in STUDIES.items():
        step = spec['step']
        lines += [f'**{study.capitalize()}** (final step {step // 1000}k; held-out L1_own / goal gap; latest Tier A step '
                  'in brackets while a run trains):', '',
                  '| Seed | Task | ' + ' | '.join(f'{r} L1_own' for r in ROLES) + ' | ' + ' | '.join(f'{r} gap' for r in ROLES) + ' |',
                  '| --- | --- |' + ' ---: |' * 6]
        for seed, runs in spec['runs'].items():
            results = {role: tier_a(runs_root, run, step) for role, run in runs.items()}
            for task, short in TASKS.items():
                if study == 'single-task' and task != RELOC:
                    continue
                cells = [cell(runs_root, runs[role], results[role], task, key) for key in ('L1_own', 'gap') for role in ROLES]
                lines.append(f'| s{seed} | {short} | ' + ' | '.join(cells) + ' |')
            final[(study, seed)] = results
        lines.append('')
    complete = all(all(r is not None for r in results.values()) for results in final.values())
    if not complete:
        lines.append('Reading: pending (every final result is needed).')
        return '\n'.join(lines), None
    single = {seed: tuple(metric(final[('single-task', seed)][role], RELOC, 'L1_own') for role in ROLES) for seed in (0, 1)}
    label, text = reading(single)
    lines += [f'**Reading (single-task, primary): {label}.** {text}', '']
    raised = []
    for task, short in TASKS.items():
        pairs = [(metric(final[('four-task', s)]['control'], task, 'gap'), metric(final[('four-task', s)]['head-only'], task, 'gap'))
                 for s in (0, 1)]
        up = all(c is not None and h is not None and c > h for c, h in pairs)
        raised.append(f'{short}: control gap {"above" if up else "not above"} head-only at both seeds '
                      f'({", ".join(f"{c:.4f} vs {h:.4f}" for c, h in pairs)})')
    lines += ['**Four-task goal reliance (control vs head-only):** ' + '; '.join(raised) + '. Where the control raises reliance '
              'on dishwasher and the board-game control too, that part of the wave-2 pattern comes from the shared filters, '
              'not wrist content.', '']
    head = [(metric(final[(study, s)]['control'], RELOC, 'gap'), metric(final[(study, s)]['head-only'], RELOC, 'gap'))
            for study in STUDIES for s in (0, 1)]
    lines.append('**Head-goal reliance on relocalization (control gap vs head-only gap; single-task s0, s1, four-task s0, s1):** '
                 + ', '.join(f'{c:.4f} vs {h:.4f}' for c, h in head)
                 + ('; higher in every pair: extra shared-filter training raises head-goal use.' if all(c > h for c, h in head)
                    else '; not higher in every pair.'))
    return '\n'.join(lines), f'Wrist-goal control, final reading (single-task, primary): {label}. {text}'
