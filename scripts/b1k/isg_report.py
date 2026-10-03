#!/usr/bin/env python3
"""Markdown tables of an ISG wave (training progress, uploads, Tier A, wave analysis, bf16 check) for the report.

    isg_report.py MANIFEST [--report /tmp/dev/report.md]

Tier A comes from the manifest's `tier_a` result directory (the held-out test set when the manifest names one);
results on the training dataset's own held-out episodes (`tierA/`) are kept in a collapsed legacy table. The wave
analysis applies the plan's decision standards (docs/isg-goal-conditioning-plan.md section 5) to the Tier A results.

Without --report the tables go to stdout; with it they replace the text between `<!-- isg-runs:begin -->` and
`<!-- isg-runs:end -->` in that file (appended once if the markers are missing).
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isg_tier_a_watch import SHORT  # noqa: E402
from isg_wave import LEGACY_TIER_A, load, max_steps, paths, repo_id, tier_a_config  # noqa: E402

BEGIN, END = '<!-- isg-runs:begin -->', '<!-- isg-runs:end -->'


def records(run_dir):
    path = run_dir / 'metrics.jsonl'
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else float('nan')


def fmt(value, digits=4):
    return '—' if value is None or value != value else f'{value:.{digits}f}'


def progress_table(manifest):
    lines = ['| Run | Role | GPU | Step | s/step | Peak GiB | Train L1 (last 500) | ETA h | Hub eval steps | Hub resume step | Exit |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |']
    for name, run in manifest['_runs'].items():
        where = paths(manifest, name)
        rows = records(where['run'])
        if not rows:
            continue
        last = rows[-1]
        recent = rows[-500:]
        times = sorted(r['timing/step_s'] for r in rows[-200:])
        step_s = times[len(times) // 2]
        exit_code = where['exit'].read_text().strip() if where['exit'].exists() else 'running'
        eta = (max_steps(manifest, run) - last['step']) * step_s / 3600 if exit_code == 'running' else float('nan')
        hub_eval, hub_full = '—', '—'
        status = where['staging'] / 'status.json'
        if status.exists():
            value = json.loads(status.read_text())
            hub_eval = ', '.join(f'{s // 1000}k' for s in value['eval_steps']) or 'none yet'
            hub_full = str(value['current_step'])
        role = run.get('role', '').split(';')[0].split('(')[0].strip()
        lines.append(f'| `{name}` | {role} | {run.get("gpu", "—")} | {last["step"]} | {step_s:.3f} | '
                     f'{max(r.get("gpu/peak_reserved_bytes", 0) for r in rows) / 2**30:.0f} | {mean(recent, "l1"):.4f} | '
                     f'{fmt(eta, 1)} | {hub_eval} | {hub_full} | {exit_code} |')
    return lines


GOAL_TASKS = ('reloc', 'dishwasher', 'blender')
ALL_TASKS = (*GOAL_TASKS, 'boardgame')
TASK_OF = {short: task for task, short in SHORT.items()}


def tier_a_results(manifest, directory):
    """{run: {step: result}} of every Tier A JSON in <run>/<directory>/."""
    results = {}
    for name in manifest['_runs']:
        folder = paths(manifest, name)['run'] / directory
        for path in sorted(folder.glob('step_*.json')) if folder.exists() else []:
            if not path.stem.endswith('.probe'):
                result = json.loads(path.read_text())
                results.setdefault(name, {})[int(result['step'])] = result
    return results


def task_entry(result, subset, short):
    return (result.get(subset) or {}).get('tasks', {}).get(TASK_OF[short])


def goal_mean(result, subset, key):
    """Mean of `key` over the goal-reading tasks (relocalization, dishwasher, blender) and its standard error."""
    entries = [task_entry(result, subset, short) for short in GOAL_TASKS]
    entries = [e for e in entries if e and e.get(key) is not None]
    if not entries:
        return None, None
    ses = [e.get(f'{key}_se') or 0.0 for e in entries]
    return sum(e[key] for e in entries) / len(entries), sum(v * v for v in ses) ** 0.5 / len(entries)


def tier_a_tables(manifest):
    config = tier_a_config(manifest)
    results = tier_a_results(manifest, config['dir'])
    if not results:
        return [f'No Tier A results in `<run>/{config["dir"]}/` yet (first checkpoint at 10k).']
    heldout = config['heldout_dataset']
    lines = [(f'Held-out = `{heldout}` (`isg_meta/eval_split.json`): the test-set instances, 50 per task for all four '
              'tasks, collected on instances no training demo uses. ' if heldout else
              'Held-out = the training dataset\'s own held-out episodes. ') +
             'Train = the 100 training episodes per task. Normalized action units, 256 frames per task and subset '
             '(the same frames for every checkpoint). `L1_own`, `L1_masked` and `swap` are means over the four tasks; '
             '`gap` (goal tasks) is the mean over relocalization, dishwasher and blender, with board game (the '
             'label-only control, where the goal carries no information beyond the task) in its own column. '
             '`±` is one standard error of the probe\'s frame sample (not seed noise).', '',
             '| Run | Step | held-out L1_own | held-out gap (goal tasks) | held-out gap boardgame | held-out L1_masked | '
             'held-out swap | train L1_own | train gap (goal tasks) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    detail = ['', 'Per task (held-out), `L1_own / gap / swap`:', '',
              '| Run | Step | ' + ' | '.join(ALL_TASKS) + ' |', '| --- | ---: | ' + ' | '.join('---' for _ in ALL_TASKS) + ' |']
    for name, by_step in results.items():
        for step, result in sorted(by_step.items()):
            held, train = result['heldout'].get('mean', {}), result['train'].get('mean', {})
            gap, gap_se = goal_mean(result, 'heldout', 'gap')
            board = task_entry(result, 'heldout', 'boardgame')
            board_gap = f'{fmt(board["gap"])} ± {fmt(board.get("gap_se"))}' if board else '—'
            train_gap, _ = goal_mean(result, 'train', 'gap')
            lines.append(f'| `{name}` | {step} | {fmt(held.get("L1_own"))} | {fmt(gap)} ± {fmt(gap_se)} | {board_gap} | '
                         f'{fmt(held.get("L1_masked"))} | {fmt(held.get("swap"))} | {fmt(train.get("L1_own"))} | '
                         f'{fmt(train_gap)} |')
            cells = []
            for short in ALL_TASKS:
                entry = task_entry(result, 'heldout', short)
                cells.append('—' if not entry else f'{fmt(entry["L1_own"])} / {fmt(entry["gap"])} / {fmt(entry["swap"], 3)}')
            detail.append(f'| `{name}` | {step} | ' + ' | '.join(cells) + ' |')
    return lines + detail


def legacy_tier_a_table(manifest):
    """Collapsed table of the results on the training dataset's own held-out episodes, if a test set replaced them."""
    if tier_a_config(manifest)['dir'] == LEGACY_TIER_A['dir']:
        return []
    results = tier_a_results(manifest, LEGACY_TIER_A['dir'])
    if not results:
        return []
    lines = ['<details><summary>Legacy Tier A: isg-init\'s own held-out episodes (26 relocalization, 5 dishwasher, '
             '8 blender, no board game; the basis of the 10k goal-LR decision)</summary>', '',
             '| Run | Step | held-out L1_own | held-out gap | held-out L1_masked | held-out swap | train L1_own | train gap |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, by_step in results.items():
        for step, result in sorted(by_step.items()):
            held, train = result['heldout'].get('mean', {}), result['train'].get('mean', {})
            lines.append(f'| `{name}` | {step} | {fmt(held.get("L1_own"))} | {fmt(held.get("gap"))} | '
                         f'{fmt(held.get("L1_masked"))} | {fmt(held.get("swap"))} | {fmt(train.get("L1_own"))} | '
                         f'{fmt(train.get("gap"))} |')
    return lines + ['', '</details>']


def slots(manifest):
    """{slot number: run name} from the manifest roles that start with 'wave-1 slot N'."""
    found = {}
    for name, run in manifest['_runs'].items():
        match = re.match(r'wave-\d+ slot (\d+)', run.get('role', ''))
        if match:
            found[int(match.group(1))] = name
    return dict(sorted(found.items()))


def metric(result, key, task=None):
    """Held-out task mean (`task` None), goal-task gap mean (`key` 'gap_goal') or one task's value."""
    if result is None:
        return None
    if key == 'gap_goal':
        return goal_mean(result, 'heldout', 'gap')[0]
    if task:
        entry = task_entry(result, 'heldout', task)
        return entry.get(key) if entry else None
    return result['heldout'].get('mean', {}).get(key)


def cost(manifest, name):
    rows = records(paths(manifest, name)['run'])
    if not rows:
        return None, None
    times = sorted(r['timing/step_s'] for r in rows[-2000:])
    return times[len(times) // 2], max(r.get('gpu/peak_reserved_bytes', 0) for r in rows) / 2 ** 30


def wave_analysis(manifest):
    """Plan section 5 on the Tier A results: seed noise floor, 20k kill rule, 50k ranking and persistence."""
    config = tier_a_config(manifest)
    results = tier_a_results(manifest, config['dir'])
    slot = slots(manifest)
    final = max_steps(manifest, {})
    if 1 not in slot or 2 not in slot:
        return ['Wave analysis needs slots 1 and 2 (baseline and its seed replicate).']
    base, rep = results.get(slot[1], {}), results.get(slot[2], {})
    steps = [s for s in (final - 10000, final) if s in base and s in rep]
    if len(steps) < 2:
        return [f'Wave analysis waits for Tier A of slots 1 and 2 at {final - 10000} and {final}.']
    keys = [('L1_own', None), ('gap_goal', None), ('L1_masked', None), ('swap', None)]
    keys += [('gap', t) for t in ALL_TASKS] + [('L1_own', t) for t in ALL_TASKS]

    def label(key, task):
        return f'{task} {key}' if task else {'gap_goal': 'gap (goal tasks)'}.get(key, key)

    delta = {}
    lines = ['#### Noise floor (slot 1 vs slot 2: same configuration, seeds 0 and 1)', '',
             '| Metric | ' + ' | '.join(f'seed 0 @ {s // 1000}k | seed 1 @ {s // 1000}k' for s in steps) + ' | δ |',
             '| --- | ' + ' | '.join('---: | ---:' for _ in steps) + ' | ---: |']
    for key, task in keys:
        values = [(metric(base[s], key, task), metric(rep[s], key, task)) for s in steps]
        if any(a is None or b is None for a, b in values):
            continue
        floor = 0.003 * abs(values[-1][0]) if key.startswith('L1') else 0.0
        delta[(key, task)] = max(max(abs(a - b) for a, b in values), floor)
        lines.append(f'| {label(key, task)} | ' + ' | '.join(f'{fmt(a)} | {fmt(b)}' for a, b in values) +
                     f' | {fmt(delta[(key, task)])} |')
    lines += ['', 'δ = the larger |seed 0 − seed 1| of the two steps (for L1 metrics at least 0.3 % of the baseline, the '
              'plan\'s dropout noise floor). One seed pair is a single draw of seed noise, so δ is a rough scale.', '']

    # 20k kill rule, applied with the final noise floor
    lines += ['#### 20k triage (kill rule)', '',
              '| Slot | Run | L1_own @ 20k | × baseline | gap (goal tasks) @ 20k | s/step | Kill? |',
              '| ---: | --- | ---: | ---: | ---: | ---: | --- |']
    b20 = metric(base.get(20000), 'L1_own')
    for number, name in slot.items():
        r20 = results.get(name, {}).get(20000)
        own, gap = metric(r20, 'L1_own'), metric(r20, 'gap_goal')
        step_s, _ = cost(manifest, name)
        if own is None or b20 is None:
            lines.append(f'| {number} | `{name}` | — | — | — | — | no 20k result |')
            continue
        kill = own >= 1.10 * b20 and gap <= delta[('gap_goal', None)]
        lines.append(f'| {number} | `{name}` | {fmt(own)} | {own / b20:.2f} | {fmt(gap)} | {fmt(step_s, 3)} | '
                     f'{"**kill**" if kill else "keep"} |')

    # 50k ranking
    lines += ['', f'#### {final // 1000}k ranking (held-out L1_own; flags from the noise floor)', '',
              '| Rank | Slot | Run | L1_own | Δ vs baseline (δ units) | gap reloc | gap dishwasher | gap blender | '
              'gap boardgame | Reads the goal? | s/step | GiB |',
              '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    rows = []
    for number, name in slot.items():
        result = results.get(name, {}).get(final)
        if result is None:
            continue
        gaps = {t: metric(result, 'gap', t) for t in ALL_TASKS}
        blind = [t for t in GOAL_TASKS if gaps[t] is not None and gaps[t] <= delta[('gap', t)]]
        rows.append((bool(blind), metric(result, 'L1_own'), number, name, gaps, blind))
    rows.sort(key=lambda row: (row[0], row[1]))
    b50 = metric(base[final], 'L1_own')
    d_own = delta[('L1_own', None)]
    for rank, (_, own, number, name, gaps, blind) in enumerate(rows, 1):
        step_s, gib = cost(manifest, name)
        reads = 'yes' if not blind else 'no: gap ≤ δ on ' + ', '.join(blind)
        lines.append(f'| {rank} | {number} | `{name}` | {fmt(own)} | {(own - b50) / d_own:+.1f} | ' +
                     ' | '.join(fmt(gaps[t]) for t in ALL_TASKS) + f' | {reads} | {fmt(step_s, 3)} | {gib:.0f} |')
    lines += ['', 'Ranking rule (plan section 5): held-out `L1_own`, with any run whose `gap` is within δ of zero on '
              'relocalization, dishwasher or blender ranked below every run that reads the goal on all three. '
              'Board game is the control: a large `gap` there is a warning, not a merit.', '']

    # persistence against the baseline
    lines += ['#### Persistence against the baseline (difference must exceed 2δ at both steps with the same sign)', '',
              '| Slot | Run | ' + ' | '.join(f'Δ L1_own @ {s // 1000}k' for s in steps) + ' | L1_own verdict | ' +
              ' | '.join(f'Δ gap (goal tasks) @ {s // 1000}k' for s in steps) + ' | gap verdict |',
              '| ---: | --- | ' + ' | '.join('---:' for _ in steps) + ' | --- | ' + ' | '.join('---:' for _ in steps) +
              ' | --- |']

    def verdict(diffs, d, lower_is_better):
        if all(x > 2 * d for x in diffs) or all(x < -2 * d for x in diffs):
            better = (diffs[0] < 0) == lower_is_better
            return 'better' if better else 'worse'
        return 'tie'
    for number, name in slot.items():
        if number == 1:
            continue
        run = results.get(name, {})
        if not all(s in run for s in steps):
            continue
        d1 = [metric(run[s], 'L1_own') - metric(base[s], 'L1_own') for s in steps]
        d2 = [metric(run[s], 'gap_goal') - metric(base[s], 'gap_goal') for s in steps]
        lines.append(f'| {number} | `{name}` | ' + ' | '.join(f'{x:+.4f}' for x in d1) +
                     f' | {verdict(d1, d_own, True)} | ' + ' | '.join(f'{x:+.4f}' for x in d2) +
                     f' | {verdict(d2, delta[("gap_goal", None)], False)} |')
    lines += ['', f'2δ: L1_own {fmt(2 * d_own)}, gap (goal tasks) {fmt(2 * delta[("gap_goal", None)])}. Slot 2 is the '
              'baseline\'s own seed replicate, so it is a tie by construction of δ.', '']
    return lines + paired_by_seed(manifest, results, slot, final)


def paired_by_seed(manifest, results, slot, final):
    """Each mechanism minus the baseline of the same seed (runs of one seed share their initialization), from the
    'seed-N replicate of wave-1 slot K' runs; and the baseline's spread over its seeds."""
    seeds = {}  # (slot, seed) -> run
    for number, name in slot.items():
        seeds[(number, int(manifest['_runs'][name]['seed']))] = name
    for name, run in manifest['_runs'].items():
        match = re.match(r'seed-(\d+) replicate of wave-\d+ slot (\d+)', run.get('role', ''))
        if match:
            seeds[(int(match.group(2)), int(match.group(1)))] = name
    baselines = {seed: name for (number, seed), name in seeds.items() if number == 1 and seed != 0}
    baselines[0] = slot[1]
    baselines.update({1: slot[2]} if 2 in slot else {})
    replicated = sorted({number for (number, seed) in seeds if number not in (1, 2) and seed != 0})
    if not replicated and len(baselines) < 3:
        return []
    lines = [f'#### Paired by initialization (each run minus the baseline of the same seed, {final // 1000}k)', '']
    finals = {seed: results.get(name, {}).get(final) for seed, name in baselines.items()}
    done = {seed: finals[seed] for seed in sorted(finals) if finals[seed] is not None}
    if len(done) >= 3:
        own = [metric(result, 'L1_own') for result in done.values()]
        gap = [metric(result, 'gap_goal') for result in done.values()]
        lines += [f'Baseline over seeds {sorted(done)}: L1_own ' + ' / '.join(fmt(v) for v in own) +
                  f' (range {fmt(max(own) - min(own))}), gap (goal tasks) ' + ' / '.join(fmt(v) for v in gap) +
                  f' (range {fmt(max(gap) - min(gap))}).', '']
    header = ['| Slot | Seed | Run | Δ L1_own | Δ gap (goal tasks) | Δ gap reloc | Δ gap boardgame |',
              '| ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    rows, waiting = [], []
    for number in replicated:
        for seed in sorted(s for (n, s) in seeds if n == number):
            name, base = seeds[(number, seed)], finals.get(seed)
            result = results.get(name, {}).get(final)
            if result is None or base is None:
                waiting.append(f'`{name}`')
                continue
            rows.append(f'| {number} | {seed} | `{name}` | {metric(result, "L1_own") - metric(base, "L1_own"):+.4f} | '
                        f'{metric(result, "gap_goal") - metric(base, "gap_goal"):+.4f} | '
                        f'{metric(result, "gap", "reloc") - metric(base, "gap", "reloc"):+.4f} | '
                        f'{metric(result, "gap", "boardgame") - metric(base, "gap", "boardgame"):+.4f} |')
    if rows:
        lines += header + rows + ['', 'A mechanism difference that holds for both initializations (same sign, similar '
                                  'size) is the evidence the promotion rule asks for (two seeds agree).']
    if waiting:
        lines += ['', f'Waiting for the {final // 1000}k Tier A of ' + ', '.join(waiting) + ' (or of its baseline).']
    return lines


def bf16_table(manifest):
    tf32 = [n for n, r in manifest['_runs'].items() if n.startswith('pf-bf16check')]
    lines = []
    for name in tf32:
        arm = manifest['_runs'][name]
        bf16_name = name.replace('pf-bf16check-', 'w1-').replace('-tf32', '')
        a, b = records(paths(manifest, name)['run']), records(paths(manifest, bf16_name)['run'])
        window = lambda rows: [r for r in rows if 501 <= r['step'] <= 1000]  # noqa: E731
        wa, wb = window(a), window(b)
        if len(wa) < 500 or len(wb) < 500:
            lines.append(f'bf16 check pending ({len(wa)}/500 TF32 steps, {len(wb)}/500 bf16 steps in 501–1000).')
            continue
        lines += ['| Arm | run | mean L1 (501–1000) | mean KL | mean loss | median s/step | peak GiB |',
                  '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for label, run_name, rows in (('TF32', name, wa), ('bf16-backbone', bf16_name, wb)):
            times = sorted(r['timing/step_s'] for r in rows)
            lines.append(f'| {label} | `{run_name}` | {mean(rows, "l1"):.5f} | {mean(rows, "kl"):.5f} | '
                         f'{mean(rows, "loss"):.5f} | {times[len(times) // 2]:.3f} | '
                         f'{max(r.get("gpu/peak_reserved_bytes", 0) for r in rows) / 2**30:.0f} |')
        rel = (mean(wb, 'l1') - mean(wa, 'l1')) / mean(wa, 'l1') * 100
        lines.append(f'\nRelative L1 difference (bf16 − TF32) / TF32 = {rel:+.2f} % (noise floor from the plan: ±0.3 %).')
    return lines


def render(manifest):
    now = datetime.now(timezone(timedelta(hours=-7), 'PDT')).strftime('%Y-%m-%d %H:%M %Z')  # host has no tzdata
    out = [BEGIN, f'_Tables regenerated {now} by `scripts/b1k/isg_report.py`._', '', '### Training progress', '',
           *progress_table(manifest), '', 'Hub repos: ' + ', '.join(
               f'[`{repo_id(manifest, n)}`](https://huggingface.co/{repo_id(manifest, n)})'
               for n, r in manifest['_runs'].items() if r.get('upload', True) and paths(manifest, n)['staging'].exists()),
           '', '### Tier A', '', *tier_a_tables(manifest), '', *legacy_tier_a_table(manifest), '',
           '### Wave analysis', '', *wave_analysis(manifest), '', '### bf16 check (pre-flight item 4)', '',
           *bf16_table(manifest), END]
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('manifest')
    parser.add_argument('--report')
    args = parser.parse_args()
    text = render(load(args.manifest))
    if not args.report:
        print(text)
        return
    report = Path(args.report)
    current = report.read_text() if report.exists() else ''
    if BEGIN in current and END in current:
        current = current[:current.index(BEGIN)] + text + current[current.index(END) + len(END):]
    else:
        current = current.rstrip('\n') + '\n\n' + text + '\n'
    report.write_text(current)


if __name__ == '__main__':
    main()
