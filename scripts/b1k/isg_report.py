#!/usr/bin/env python3
"""Markdown tables of an ISG wave (training progress, uploads, Tier A, bf16 check) for the overnight report.

    isg_report.py MANIFEST [--report /tmp/dev/report.md]

Without --report the tables go to stdout; with it they replace the text between `<!-- isg-runs:begin -->` and
`<!-- isg-runs:end -->` in that file (appended once if the markers are missing).
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isg_tier_a_watch import SHORT  # noqa: E402
from isg_wave import load, max_steps, paths, repo_id  # noqa: E402

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
        eta = (max_steps(manifest, run) - last['step']) * step_s / 3600
        exit_code = where['exit'].read_text().strip() if where['exit'].exists() else 'running'
        hub_eval, hub_full = '—', '—'
        status = where['staging'] / 'status.json'
        if status.exists():
            value = json.loads(status.read_text())
            hub_eval = ', '.join(f'{s // 1000}k' for s in value['eval_steps']) or 'none yet'
            hub_full = str(value['current_step'])
        role = run.get('role', '').split(';')[0].split('(')[0].strip()
        lines.append(f'| `{name}` | {role} | {run.get("gpu", "—")} | {last["step"]} | {step_s:.3f} | '
                     f'{max(r.get("gpu/peak_reserved_bytes", 0) for r in rows) / 2**30:.0f} | {mean(recent, "l1"):.4f} | '
                     f'{eta:.1f} | {hub_eval} | {hub_full} | {exit_code} |')
    return lines


def tier_a_tables(manifest):
    lines = ['Task means; held-out = relocalization, dishwasher open, blender open (board game has no held-out episodes '
             'today); train adds board game. Normalized action units, 256 frames per task and subset; `±` is one '
             'standard error of the probe\'s frame sample (not seed noise).', '',
             '| Run | Step | held-out L1_own | gap_heldout | held-out L1_masked | swap (held-out) | train L1_own | gap_train | '
             'swap (train) |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    detail = ['', 'Per task (held-out except board game, which is train):', '',
              '| Run | Step | ' + ' | '.join(f'{t} L1_own / gap / swap' for t in ('reloc', 'dishwasher', 'blender', 'boardgame (train)')) + ' |',
              '| --- | ---: | ' + ' | '.join('---' for _ in range(4)) + ' |']
    any_result = False
    for name in manifest['_runs']:
        directory = paths(manifest, name)['run'] / 'tierA'
        if not directory.exists():
            continue
        for path in sorted(directory.glob('step_*.json')):
            if path.stem.endswith('.probe'):
                continue
            result = json.loads(path.read_text())
            any_result = True
            held, train = result['heldout'].get('mean', {}), result['train'].get('mean', {})

            def se_of(subset, key):
                tasks = result[subset].get('tasks', {})
                values = [entry.get(f'{key}_se') for entry in tasks.values() if entry.get(f'{key}_se') is not None]
                return (sum(v * v for v in values) ** 0.5 / len(values)) if values else None
            gap_se = se_of('heldout', 'gap')
            lines.append(f'| `{name}` | {result["step"]} | {fmt(held.get("L1_own"))} | {fmt(held.get("gap"))} ± '
                         f'{fmt(gap_se)} | {fmt(held.get("L1_masked"))} | {fmt(held.get("swap"))} | '
                         f'{fmt(train.get("L1_own"))} | {fmt(train.get("gap"))} | {fmt(train.get("swap"))} |')
            cells = []
            for task, short in [(t, s) for t, s in SHORT.items() if s in ('reloc', 'dishwasher', 'blender', 'boardgame')]:
                subset = 'train' if short == 'boardgame' else 'heldout'
                entry = result[subset].get('tasks', {}).get(task)
                cells.append('—' if not entry else f'{fmt(entry["L1_own"])} / {fmt(entry["gap"])} / {fmt(entry["swap"], 3)}')
            detail.append(f'| `{name}` | {result["step"]} | ' + ' | '.join(cells) + ' |')
    if not any_result:
        return ['No Tier A results yet (first checkpoint at 10k).']
    return lines + detail


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
           '', '### Tier A', '', *tier_a_tables(manifest), '', '### bf16 check (pre-flight item 4)', '',
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
