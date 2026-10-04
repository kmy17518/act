#!/usr/bin/env python3
"""Machine 1's collector for the ISG goal study's helper machines (plan §7, coordination repo).

Every cycle (default every 30 minutes):
  1. download `m2/` and `lab/` of the coordination repo into /tmp/dev/coord/in/ (a fresh copy each cycle);
  2. copy machine 2's Tier A results `m2/tierA-eval50/<run>/step_*.json` into `<runs_root>/<run>/tierA-eval50/`, only
     for runs that `waves/wave2.json` assigns to "M2" and that have no local training log;
  3. render the report for `waves/wave1.json` and `waves/wave2.json` plus a "Machine 2 and lab" section at its top
     (helper `needs_decision` / `blocked` items first);
  4. apply machine 1's rules and record every decision in `m1/decisions.md` and the plan's decision log: the machine-2
     calibration criterion, the plan §5 20k kill rule on machine-2 runs, the Tier B trust conditions (`instances.json`
     covering all 200 test episodes with verified first frames, one fixed action horizon in `env.json`) and, once
     every queued finalist has a `summary.json`, the plan §5 wave verdict (paired bootstrap over the instances);
  5. flag a helper whose `status.json` is more than an hour old (machine 2: propose the plan §7 recovery, never do it);
  6. upload `docs/` and `m1/` in one commit when anything changed.
It never writes under `m2/` or `lab/`, never starts uploaders, and writes locally only under /tmp.

    isg_collect.py [--interval 1800] [--once]
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

CHECKOUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHECKOUT / 'scripts/b1k'))
from isg_wave import load, tier_a_config  # noqa: E402

REPO = 'kmy17518/isg-goal-coordination'
COORD = Path('/tmp/dev/coord')
PLAN = Path('/tmp/dev/docs/isg-goal-conditioning-plan.md')
REPORT = Path('/tmp/dev/report.md')
CALIBRATION_REFERENCE, CALIBRATION_TOLERANCE = 0.12560, 0.003
STALE = timedelta(hours=1)
# lab/instances.json: a mapping counts as verified when its first head frame matches the test episode's first frame to
# within this mean absolute difference on the 0-255 scale (values that are all <= 1 are read as the 0-1 scale)
FIRST_FRAME_MAX_DIFF = 20.0
REFERENCE_KEYS = ('ref', 'machine1', 'machine_1', 'expected', 'target', 'baseline', 'm1_')
GOAL_TASKS = ('camera_relocalization-standard', 'configuration_matching-articulation_open_large_scale-dishwasher',
              'configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table')
CONTROL_TASK = 'alignment-axial-board_game-breakfast_table'
SHORT = {GOAL_TASKS[0]: 'reloc', GOAL_TASKS[1]: 'dishwasher', GOAL_TASKS[2]: 'blender', CONTROL_TASK: 'board game'}
PDT = timezone(timedelta(hours=-7), 'PDT')  # the host has no tzdata


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def iso(moment):
    return moment.isoformat().replace('+00:00', 'Z')


def parse_time(value):
    try:
        moment = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def fmt(value, digits=4):
    return '—' if not isinstance(value, (int, float)) or value != value else f'{value:.{digits}f}'


# ---------------------------------------------------------------------------------------------- inputs

def download(api, inbox):
    """Fresh copy of the helpers' prefixes (deleted remote files disappear locally too)."""
    from huggingface_hub import snapshot_download
    fresh = inbox.with_name(inbox.name + '.tmp')
    shutil.rmtree(fresh, ignore_errors=True)
    fresh.mkdir(parents=True)
    snapshot_download(REPO, repo_type='dataset', allow_patterns=['m2/**', 'lab/**'], local_dir=fresh,
                      token=os.environ.get('HF_TOKEN'))
    shutil.rmtree(fresh / '.cache', ignore_errors=True)
    shutil.rmtree(inbox, ignore_errors=True)
    fresh.rename(inbox)


def status_runs(status):
    """Machine 2's per-run records as {name: record}, from a dict keyed by run or a list of records."""
    runs = (status or {}).get('runs')
    if runs is None:  # any top-level mapping of run records
        runs = next((v for v in (status or {}).values() if isinstance(v, dict) and v and
                     all(isinstance(r, dict) and ('state' in r or 'step' in r) for r in v.values())), {})
    if isinstance(runs, list):
        runs = {r.get('name') or r.get('run'): r for r in runs if isinstance(r, dict)}
    return {name: record for name, record in runs.items() if isinstance(record, dict)}


def copy_m2_tier_a(wave2, inbox):
    copied = []
    for name, run in wave2['_runs'].items():
        if run.get('machine') != 'M2':
            continue
        source = inbox / 'm2/tierA-eval50' / name
        run_dir = Path(wave2['runs_root']) / name
        if not source.is_dir():
            continue
        if (run_dir / 'metrics.jsonl').exists():
            print(json.dumps({'event': 'skip_local_run', 'run': name}), flush=True)  # never touch a local run
            continue
        target = run_dir / tier_a_config(wave2)['dir']
        for path in sorted(source.glob('step_*.json')):
            destination = target / path.name
            if digest(destination) != digest(path):
                target.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                copied.append(f'{name}/{path.name}')
    return copied


def tier_a(manifest, run, step):
    return read_json(Path(manifest['runs_root']) / run / tier_a_config(manifest)['dir'] / f'step_{step:08d}.json')


def goal_gap(result):
    tasks = (result or {}).get('heldout', {}).get('tasks', {})
    values = [tasks[t]['gap'] for t in GOAL_TASKS if t in tasks and tasks[t].get('gap') is not None]
    return sum(values) / len(values) if values else None


def l1_own(result):
    return ((result or {}).get('heldout', {}).get('mean') or {}).get('L1_own')


def local_training(run_dir, lo, hi):
    """Mean training L1 and median s/step of a local run over steps lo..hi (last record per step wins)."""
    path = Path(run_dir) / 'metrics.jsonl'
    if not path.exists():
        return None, None
    records = {}
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if lo <= record['step'] <= hi:
                records[record['step']] = record
    if not records:
        return None, None
    times = sorted(r['timing/step_s'] for r in records.values())
    return sum(r['l1'] for r in records.values()) / len(records), times[len(times) // 2]


def noise_floor(wave1):
    """δ of plan §5 on the test set: max |seed 0 - seed 1| of the late tag-zero baseline over 40k and 50k."""
    deltas = {'L1_own': None, 'gap': None}
    for key, getter in (('L1_own', l1_own), ('gap', goal_gap)):
        values = []
        for step in (40000, 50000):
            a, b = (getter(tier_a(wave1, f'w1-late-tagzero-lrg1e-4-s{seed}', step)) for seed in (0, 1))
            if a is not None and b is not None:
                values.append(abs(a - b))
        deltas[key] = max(values) if values else None
    return deltas


def find_number(value, prefer=('501', 'mean')):
    """Machine 2's own value: the first number under a key mentioning l1 (preferring one that also mentions 501), skipping
    anything under a key that names machine 1's reference (ref, expected, target, baseline, machine1, ...)."""
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, path + [str(key).lower()])
        elif isinstance(node, (int, float)) and not isinstance(node, bool) and path and 'l1' in path[-1] \
                and not any(marker in part for part in path for marker in REFERENCE_KEYS):
            found.append(('.'.join(path), float(node)))
    walk(value, [])
    for marker in prefer:
        for key, number in found:
            if marker in key.rsplit('.', 1)[-1]:
                return key, number
    return (found[0] if found else (None, None))


def own_verdict(value):
    """Machine 2's own pass/fail statement, if calibration.json has one (bool or PASS/FAIL string)."""
    if isinstance(value, dict):
        for key, child in value.items():
            if any(m in str(key).lower() for m in ('pass', 'result', 'verdict')) and not any(m in str(key).lower() for m in REFERENCE_KEYS):
                if isinstance(child, bool):
                    return child
                if isinstance(child, str) and child.strip().upper() in ('PASS', 'PASSED', 'FAIL', 'FAILED'):
                    return child.strip().upper().startswith('PASS')
        for child in value.values():
            verdict = own_verdict(child)
            if verdict is not None:
                return verdict
    return None


def instance_coverage(instances):
    """(episodes 0-199 mapped to an instance with a verified first frame, entries parsed, first-frame diffs) from
    lab/instances.json: `{episode_index: {task, instance, method, first_frame_mean_abs_diff}}`. Verified = the diff is
    at most FIRST_FRAME_MAX_DIFF (0-255 scale; all values <= 1 are read as the 0-1 scale), or an explicit true flag."""
    items = instances.get('episodes', instances.get('instances', instances)) if isinstance(instances, dict) else instances
    if isinstance(items, dict):
        items = [dict(v, episode_index=v.get('episode_index', k)) if isinstance(v, dict) else {} for k, v in items.items()]
    verified, diffs = set(), {}
    for item in items or []:
        if isinstance(item, dict):
            value = item.get('first_frame_mean_abs_diff')
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
                diffs[str(item.get('episode_index', item.get('episode')))] = float(value)
    scale = 255.0 if diffs and max(diffs.values()) <= 1.0 else 1.0

    def ok(node):
        if isinstance(node, dict):
            return any(ok(v) for k, v in node.items() if any(m in str(k).lower() for m in ('verif', 'match', 'ok', 'pass')))
        return node is True
    for item in items or []:
        if not isinstance(item, dict):
            continue
        try:
            episode = int(item.get('episode_index', item.get('episode')))
        except (TypeError, ValueError):
            continue
        diff = diffs.get(str(item.get('episode_index', item.get('episode'))))
        flagged = any(ok(v) if isinstance(v, dict) else v is True
                      for k, v in item.items() if any(m in str(k).lower() for m in ('verif', 'first_frame')))
        if 0 <= episode < 200 and item.get('instance') not in (None, '') and \
                (flagged or (diff is not None and diff * scale <= FIRST_FRAME_MAX_DIFF)):
            verified.add(episode)
    return len(verified), len(items or []), sorted(d * scale for d in diffs.values())


def fixed_action_horizon(env):
    """The single action horizon of env.json: every value under a key containing 'action_horizon' (at any depth) must
    be the same integer."""
    values = []

    def walk(node, key=''):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, str(k).lower())
        elif isinstance(node, list) and 'action_horizon' in key:
            for v in node:
                walk(v, key)
        elif 'action_horizon' in key and isinstance(node, int) and not isinstance(node, bool):
            values.append(node)
    walk(env or {})
    return values[0] if values and len(set(values)) == 1 else None


SCORE_KEYS = {GOAL_TASKS[0]: ('geodesic', 'distance'), GOAL_TASKS[1]: ('angle',), GOAL_TASKS[2]: ('angle',),
              CONTROL_TASK: ('align', 'error')}


def primary_score(task, score):
    """The task's continuous metric (lower is better) from the episode's `score`: a number, or the first number under
    a key naming it (relocalization geodesic distance, opening-angle error, alignment error)."""
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return float(score)
    if isinstance(score, dict):
        numbers = {k: v for k, v in score.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        for marker in SCORE_KEYS.get(task, ()):
            for key, value in numbers.items():
                if marker in key.lower() and 'bucket' not in key.lower():
                    return float(value)
    return None


def tier_b_files(inbox, run, step):
    return sorted((inbox / 'lab/tierB' / run / f'step_{step:08d}').glob('*/episode_*.json'))


def tier_b_episodes(inbox, run, step):
    """{(task, episode): (success, primary score)} of the episodes that finished without an evaluator error."""
    episodes = {}
    for path in tier_b_files(inbox, run, step):
        record = read_json(path) or {}
        success = record.get('success')
        if record.get('error') or not isinstance(success, (bool, int, float)):
            continue
        episodes[(path.parent.name, path.stem)] = (float(success), primary_score(path.parent.name, record.get('score')))
    return episodes


def paired_comparison(a, b, rng_seed=0, resamples=10000):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diff = np.array([a[k][0] - b[k][0] for k in keys])
    rng = np.random.default_rng(rng_seed)
    means = diff[rng.integers(0, len(diff), (resamples, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    tasks = {}
    for task in sorted({k[0] for k in keys}):
        d = np.array([a[k][0] - b[k][0] for k in keys if k[0] == task])
        scores = [a[k][1] - b[k][1] for k in keys if k[0] == task and a[k][1] is not None and b[k][1] is not None]
        tasks[task] = {'n': len(d), 'diff': float(d.mean()),
                       'se': float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('inf'),
                       'score_diff': float(np.mean(scores)) if scores else None,
                       'score_se': float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else None}
    a_wins = lo > 0 and all(t['diff'] >= -2 * t['se'] for t in tasks.values())
    b_wins = hi < 0 and all(t['diff'] <= 2 * t['se'] for t in tasks.values())
    return {'n': len(keys), 'diff': float(diff.mean()), 'ci': [float(lo), float(hi)], 'tasks': tasks,
            'outcome': 'a' if a_wins else 'b' if b_wins else 'tie'}


# ---------------------------------------------------------------------------------------------- rules

class Collector:
    def __init__(self, api, wave1_path, wave2_path):
        self.api = api
        self.wave1_path, self.wave2_path = wave1_path, wave2_path
        self.inbox, self.m1 = COORD / 'in', COORD / 'm1'
        self.state_path = COORD / 'state.json'
        self.state = read_json(self.state_path) or {'recorded': [], 'uploaded': {}, 'lab_trusted': False}

    def decide(self, key, audience, lines, log_line):
        """Append a decision once (keyed) to m1/decisions.md and the plan's decision log."""
        if key in self.state['recorded']:
            return False
        stamp = iso(utc_now())
        with (self.m1 / 'decisions.md').open('a') as stream:
            stream.write(f'\n## {stamp} — to {audience}\n' + ''.join(f'- {line}\n' for line in lines))
        plan = PLAN.read_text()
        row = (f'| {stamp[:10]} | coordination | {log_line} | `m1/decisions.md` entry {stamp} in '
               f'`{REPO}` | see entry |\n')
        PLAN.write_text(plan if plan.endswith('\n') else plan + '\n')
        with PLAN.open('a') as stream:
            stream.write(row)
        self.state['recorded'].append(key)
        self.decisions.append(f'{stamp} to {audience}: {log_line}')
        return True

    def health(self, name, status, now):
        if status is None:
            return 'not reporting yet', False
        updated = parse_time(status.get('updated_at'))
        if updated is None:
            return 'status.json has no valid `updated_at`', False
        age = now - updated
        down = age > STALE
        was_down = self.state.setdefault('down', {}).get(name, False)
        if down and not was_down:
            proposal = (' Recovery (plan §7, pre-approved by the user on 2026-10-03): machine 1 takes over its unfinished '
                        'runs from their Hub resume checkpoints under the same W&B ids as soon as a machine-1 GPU is free '
                        '(queued seed replicates yield). M2 must not restart any run before reading m1/decisions.md.'
                        if name == 'M2' else '')
            self.decide(f'down:{name}:{iso(updated)}', 'all',
                        [f'{name} is flagged as down: its status.json was last updated at {iso(updated)}, more than an hour '
                         f'ago.{proposal}'], f'{name} flagged as down (status.json older than 1 h)')
        elif was_down and not down:
            self.decide(f'up:{name}:{iso(updated)}', 'all', [f'{name} is reporting again (status.json {iso(updated)}).'],
                        f'{name} reporting again')
        self.state['down'][name] = down
        return f'last update {iso(updated)} ({age.total_seconds() / 60:.0f} min ago)' + (' — **DOWN**' if down else ''), down

    def calibration(self, calibration):
        if calibration is None:
            return 'not reported yet'
        key, value = find_number(calibration)
        if value is None:
            self.decide('calibration:missing-l1', 'M2',
                        ['m2/calibration.json has no mean training L1 over steps 501–1000; add it as '
                         '`mean_train_l1_501_1000` (machine 1 compares it with 0.12560, tolerance 0.3 %).'],
                        'M2 calibration: mean L1 over steps 501–1000 missing')
            return 'reported without a mean L1 over steps 501–1000 (asked M2 to add it)'
        delta = value / CALIBRATION_REFERENCE - 1
        passed = abs(delta) <= CALIBRATION_TOLERANCE
        token = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()[:12]
        claimed = own_verdict(calibration)
        if claimed is not None and claimed != passed:
            self.escalations.append(f'M2 calibration: machine 2 says {"PASS" if claimed else "FAIL"}, machine 1 reads '
                                    f'`{key}` = {value:.5f} ({delta * 100:+.2f} %) as {"pass" if passed else "fail"}; '
                                    'no decision posted, review calibration.json')
            return f'{value:.5f} (`{key}`): disagrees with machine 2\'s own verdict; escalated, no decision posted'
        verdict = (f'Calibration confirmed: mean training L1 over steps 501–1000 = {value:.5f} ({delta * 100:+.2f} % vs '
                   f'machine 1\'s {CALIBRATION_REFERENCE}, within 0.3 %). M2 may start '
                   '`w2-early-{copy025,gain,diff,wrist}-lrg1e-3-s0`.' if passed else
                   f'Calibration failed: mean training L1 over steps 501–1000 = {value:.5f} ({delta * 100:+.2f} % vs '
                   f'{CALIBRATION_REFERENCE}, outside 0.3 %). M2 must not start the wave-2 runs; machine 1 will decide '
                   'the next step after checking commit, flags and GPUs.')
        self.decide(f'calibration:{token}', 'M2', [verdict], f'M2 calibration {"passed" if passed else "failed"} '
                    f'({value:.5f}, {delta * 100:+.2f} %)')
        return f'{value:.5f} (`{key}`), {delta * 100:+.2f} % vs {CALIBRATION_REFERENCE}: **{"pass" if passed else "fail"}**'

    def kill_rule(self, wave1, wave2, runs, delta):
        """Plan §5 triage at 20k for machine-2 runs against the wave-2 base run of the same seed."""
        rows = []
        base_name = (wave2.get('base') or {}).get('runs', ['w1-early-copy05-lrg1e-3-s0'])[0]
        base = tier_a(wave1, base_name, 20000)
        base_l1_train, base_s = local_training(Path(wave1['runs_root']) / base_name, 19501, 20000)
        for name, run in wave2['_runs'].items():
            if run.get('machine') != 'M2' or name.startswith('m2-calib'):
                continue
            result = tier_a(wave2, name, 20000)
            if result is None or l1_own(base) is None:
                continue
            record = runs.get(name, {})
            l1, gap = l1_own(result), goal_gap(result)
            train_l1, s_step = record.get('train_l1_last500'), record.get('s_per_step')
            reasons = []
            if isinstance(train_l1, (int, float)) and (train_l1 != train_l1 or (base_l1_train and train_l1 > 2 * base_l1_train)):
                reasons.append(f'training L1 {train_l1:.4f} non-finite or > 2× the base run\'s {base_l1_train:.4f}')
            if l1 is not None and l1 >= 1.10 * l1_own(base) and gap is not None and delta['gap'] is not None and gap <= delta['gap']:
                reasons.append(f'held-out L1_own {l1:.4f} ≥ 1.10 × base {l1_own(base):.4f} and goal-task gap {gap:.4f} ≤ δ '
                               f'{delta["gap"]:.4f}')
            if isinstance(s_step, (int, float)) and base_s and s_step > 2 * base_s:
                reasons.append(f'{s_step:.3f} s/step is below half the expected speed ({base_s:.3f} s/step on machine 1)')
            kill = bool(reasons)
            text = (f'`{name}` at 20k: **kill** (plan §5): ' + '; '.join(reasons) + '. Stop it; its GPU waits for machine 1.'
                    if kill else
                    f'`{name}` at 20k: keep. Held-out L1_own {fmt(l1)} vs base `{base_name}` {fmt(l1_own(base))} '
                    f'(threshold {fmt(1.10 * l1_own(base))}), goal-task gap {fmt(gap)} (δ {fmt(delta["gap"])}), '
                    f'{fmt(s_step, 3)} s/step.')
            self.decide(f'kill20k:{name}', 'M2', [text], f'M2 20k triage {name}: {"kill" if kill else "keep"}')
            rows.append(text)
        return rows

    def lab_trust(self, instances, env):
        self.decide('rule:first-frame-threshold', 'lab',
                    [f'Machine 1 counts an episode → instance mapping as verified when `first_frame_mean_abs_diff` is at '
                     f'most {FIRST_FRAME_MAX_DIFF:g} on the 0–255 scale (values that are all ≤ 1 are read as the 0–1 scale). '
                     'Tier B results are trusted once all 200 test episodes meet it and env.json fixes one action horizon.'],
                    f'lab verification criterion: first-frame mean abs diff <= {FIRST_FRAME_MAX_DIFF:g} (0-255)')
        covered, parsed, diffs = instance_coverage(instances) if instances is not None else (0, 0, [])
        self.first_frame_diffs = diffs
        horizon = fixed_action_horizon(env)
        trusted = covered == 200 and horizon is not None
        reasons = []
        if covered != 200:
            reasons.append(f'instances.json verifies {covered}/200 test episodes ({parsed} entries parsed)'
                           if instances is not None else 'no instances.json yet')
        if horizon is None:
            reasons.append('env.json shows no single fixed action horizon' if env is not None else 'no env.json yet')
        if trusted != self.state.get('lab_trusted', False):
            self.state['lab_trusted'] = trusted
            self.decide(f'labtrust:{trusted}:{iso(utc_now())}', 'lab',
                        [f'Lab results are trusted from now on: instances.json verifies all 200 test episodes and '
                         f'env.json fixes the action horizon at {horizon}.' if trusted else
                         'Lab results are no longer trusted: ' + '; '.join(reasons) + '.'],
                        f'lab results {"trusted" if trusted else "not trusted"}')
        return trusted, horizon, covered, reasons

    def verdict(self, queue, trusted, wave1):
        finalists = [(e['run'], int(e['step'])) for e in (queue or {}).get('entries', [])]
        summaries = {(r, s): read_json(self.inbox / 'lab/tierB' / r / f'step_{s:08d}' / 'summary.json')
                     for r, s in finalists}
        incomplete = [r for r, s in finalists if len(tier_b_files(self.inbox, r, s)) < 200]
        if not finalists or any(v is None for v in summaries.values()) or incomplete:
            return ({'pending': 'waiting for complete results (200 episode files and a summary.json) of '
                     + ', '.join(f'`{r}`' for r in incomplete)} if incomplete and all(v is not None for v in summaries.values())
                    else None), summaries
        if not trusted:
            return {'pending': 'every finalist has a summary.json, but lab results are not trusted yet'}, summaries
        episodes = {f: tier_b_episodes(self.inbox, *f) for f in finalists}
        pairs = {}
        for a, b in combinations(finalists, 2):
            pairs[(a[0], b[0])] = paired_comparison(episodes[a], episodes[b])
        beaten = {r: False for r, _ in finalists}
        wins = {r: 0 for r, _ in finalists}
        for (a, b), result in pairs.items():
            if result and result['outcome'] == 'a':
                beaten[b] = True
                wins[a] += 1
            elif result and result['outcome'] == 'b':
                beaten[a] = True
                wins[b] += 1
        cost = {r: local_training(Path(wave1['runs_root']) / r, 1, 50000)[1] or float('inf') for r, _ in finalists}
        undefeated = sorted([r for r, _ in finalists if not beaten[r]], key=lambda r: (cost[r], r))
        winner = next((r for r, _ in finalists if wins[r] == len(finalists) - 1), None)
        text = (f'Wave-1 verdict (Tier B, {len(finalists)} finalists, paired bootstrap over the test instances): '
                + (f'`{winner}` beats every other finalist.' if winner else
                   'no finalist beats all others; undefeated, cheapest first: '
                   + ', '.join(f'`{r}` ({cost[r]:.3f} s/step)' for r in undefeated)
                   + '. The plan\'s tie rule takes the cheaper variant; equal costs leave "simpler" to the user.'))
        details = [f'`{a}` − `{b}`: success diff {r["diff"]:+.3f}, 95 % CI [{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}] over '
                   f'{r["n"]} instances → {dict(a=a, b=b, tie="tie")[r["outcome"]]}' for (a, b), r in pairs.items() if r]
        token = hashlib.sha256(json.dumps({str(k): v for k, v in pairs.items()}, sort_keys=True, default=str).encode()).hexdigest()[:12]
        self.decide(f'verdict:{token}', 'all', [text, *details], 'wave-1 verdict (Tier B): '
                    + (winner or 'tie among ' + ', '.join(undefeated)))
        family = 'late' if (winner or undefeated[0]).startswith('w1-late') else 'early'
        return {'text': text, 'details': details, 'pairs': pairs, 'family': family, 'winner': winner,
                'undefeated': undefeated}, summaries

    # ------------------------------------------------------------------------------------------ report

    def coord_section(self, now, m2, lab, health, calibration_text, kill_rows, trust, verdict, summaries, wave2,
                      copied, delta):
        trusted, horizon, covered, reasons = trust
        out = ['<!-- coord:begin -->',
               f'## Machine 2 and lab (collector, {now.astimezone(PDT):%Y-%m-%d %H:%M} PDT)', '']
        asks = []
        for name, status in (('M2', m2), ('lab', lab)):
            for key in ('needs_decision', 'blocked'):
                items = (status or {}).get(key)
                items = items if isinstance(items, list) else [items] if items else []
                asks += [f'- **{name} {key.replace("_", " ")}:** {item}' for item in items]
        asks += [f'- **machine 1 review:** {item}' for item in self.escalations]
        out += ['**Needs a decision / blocked:**', *(asks or ['- none']), '']
        out += [f'**Helper health:** M2 — {health["M2"][0]}; lab — {health["lab"][0]}.', '']
        out += ['### Machine 2', '', f'- Calibration (`m2/calibration.json`): {calibration_text}.']
        if m2:
            out.append(f'- Commit {m2.get("git_commit", "—")}, GPUs: {m2.get("gpus", m2.get("GPUs", "—"))}.')
        runs = status_runs(m2)
        if runs:
            out += ['', '| Run | State | Step / max | s/step | Train L1 (last 500) | Peak GiB | ETA h | Exit | Uploader | Tier A steps |',
                    '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |']
            for name, r in runs.items():
                uploader = r.get('uploader') if isinstance(r.get('uploader'), str) else json.dumps(r.get('uploader', '—'))
                out.append(f'| `{name}` | {r.get("state", "—")} | {r.get("step", "—")} / {r.get("max_steps", "—")} | '
                           f'{fmt(r.get("s_per_step"), 3)} | {fmt(r.get("train_l1_last500"))} | {fmt(r.get("peak_gib"), 0)} | '
                           f'{fmt(r.get("eta_h"), 1)} | {r.get("exit_code", "—")} | {uploader[:60]} | '
                           f'{r.get("tier_a_steps", r.get("tierA_steps", "—"))} |')
        rows = []
        for name, run in wave2['_runs'].items():
            if run.get('machine') != 'M2':
                continue
            for path in sorted((Path(wave2['runs_root']) / name / tier_a_config(wave2)['dir']).glob('step_*.json')):
                result = read_json(path) or {}
                base = tier_a(self.wave1, 'w1-early-copy05-lrg1e-3-s0', int(result.get('step', 0)))
                control = (result.get('heldout', {}).get('tasks', {}).get(CONTROL_TASK) or {}).get('gap')
                rows.append(f'| `{name}` | {result.get("step")} | {fmt(l1_own(result))} | {fmt(goal_gap(result))} | '
                            f'{fmt(control)} | {fmt(l1_own(result) - l1_own(base)) if l1_own(base) is not None and l1_own(result) is not None else "—"} |')
        if rows:
            out += ['', 'Machine-2 Tier A on the test set (copied from `m2/tierA-eval50/`); Δ against the wave-2 base '
                    f'`w1-early-copy05-lrg1e-3-s0` at the same step (δ L1_own {fmt(delta["L1_own"])}, δ gap {fmt(delta["gap"])}):', '',
                    '| Run | Step | L1_own | gap (goal tasks) | gap board game | Δ L1_own vs base |',
                    '| --- | ---: | ---: | ---: | ---: | ---: |', *rows]
        if kill_rows:
            out += ['', '20k triage:', *[f'- {r}' for r in kill_rows]]
        out += ['', '### Lab', '']
        if lab:
            fields = {k: v for k, v in lab.items() if k not in ('needs_decision', 'blocked') and not isinstance(v, (list, dict))}
            out.append('- Status: ' + ', '.join(f'{k} {v}' for k, v in fields.items()) + '.')
        diffs = getattr(self, 'first_frame_diffs', [])
        if diffs:
            median = diffs[len(diffs) // 2]
            out.append(f'- First-frame check: {len(diffs)} episodes, mean abs diff median {median:.2f}, max {diffs[-1]:.2f} '
                       f'(0–255 scale; threshold {FIRST_FRAME_MAX_DIFF:g}); above 3× the median: '
                       f'{sum(d > 3 * median for d in diffs)}.')
        out.append(f'- Trust: {"**trusted**" if trusted else "not trusted: " + "; ".join(reasons)} (instances verified '
                   f'{covered}/200, action horizon {horizon if horizon is not None else "—"}).')
        smoke = sorted(p.relative_to(self.inbox).as_posix() for p in (self.inbox / 'lab/smoke').rglob('*') if p.is_file()) \
            if (self.inbox / 'lab/smoke').exists() else []
        out.append(f'- Smoke files: {len(smoke)}' + (f' (`{"`, `".join(smoke[:8])}`{" …" if len(smoke) > 8 else ""})' if smoke else '') + '.')
        if any(v is not None for v in summaries.values()):
            out += ['', '| Tier B run | Step | Episodes | Summary |', '| --- | ---: | ---: | --- |']
            for (run, step), summary in summaries.items():
                count = len(tier_b_episodes(self.inbox, run, step))
                out.append(f'| `{run}` | {step} | {count} | '
                           f'{json.dumps(summary, sort_keys=True)[:300] if summary is not None else "pending"} |')
        if verdict:
            out += ['', '### Wave verdict (Tier B)', '', verdict.get('text', verdict.get('pending', ''))]
            out += [f'- {d}' for d in verdict.get('details', [])]
            out.append('- Per-task lines: success difference ± paired SE; "score" is the difference of the task\'s '
                       'continuous metric (geodesic distance, angle error or alignment error; lower is better), errored '
                       'episodes excluded from the pairs.')
            for (a, b), r in (verdict.get('pairs') or {}).items():
                if r:
                    out.append(f'- `{a}` − `{b}` per task: ' + '; '.join(
                        f'{SHORT.get(t, t)} {v["diff"]:+.2f} ± {v["se"]:.2f} (score {fmt(v["score_diff"], 2)})'
                        for t, v in r['tasks'].items()))
            if verdict.get('family'):
                out += ['', '**Proposal to the user (wave 2, verdict-dependent):** ' + (
                    'early fusion won: keep the running early slate; next, second seeds of the best wave-2 refinements.'
                    if verdict['family'] == 'early' else
                    'late fusion won: let the early refinements finish, then run the late slate (`--goal-entry decoder`, '
                    '`--goal-entry queries`, `--goal-content diff`, `--goal-pos none`, late winner seed 2).')]
        out += ['', f'Copied this cycle: {len(copied)} machine-2 Tier A files. Decisions this cycle: '
                + ('; '.join(self.decisions) if self.decisions else 'none') + '.', '<!-- coord:end -->']
        return '\n'.join(out)

    def render_report(self, coord_text):
        from isg_report import BEGIN, END, render
        text = REPORT.read_text()
        sections = [(coord_text, '<!-- coord:begin -->', '<!-- coord:end -->', 'top')]
        for path, begin, end in ((self.wave1_path, BEGIN, END),
                                 (self.wave2_path, '<!-- isg-runs-wave2:begin -->', '<!-- isg-runs-wave2:end -->')):
            try:
                body = render(load(path))
                body = body.replace(BEGIN, begin).replace(END, end)
                if begin != BEGIN:
                    body = body.replace(begin, begin + '\n## Wave 2 tables (machine 1; machine-2 Tier A copied from the '
                                        'coordination repo)', 1)
            except Exception as exc:  # keep the cycle alive; show the failure in the report
                body = f'{begin}\n_Rendering {path.name} failed: {type(exc).__name__}: {exc}_\n{end}'
            sections.append((body, begin, end, 'end'))
        for body, begin, end, place in sections:
            if begin in text and end in text:
                text = text[:text.index(begin)] + body + text[text.index(end) + len(end):]
            elif place == 'top':
                first, _, rest = text.partition('\n')
                text = first + '\n\n' + body + '\n\n' + rest
            else:
                text = text.rstrip('\n') + '\n\n' + body + '\n'
        temporary = REPORT.with_name(REPORT.name + '.tmp')
        temporary.write_text(text)
        temporary.replace(REPORT)

    # ------------------------------------------------------------------------------------------ cycle

    def upload(self):
        from huggingface_hub import CommitOperationAdd
        files = {'docs/plan.md': PLAN, 'docs/report.md': REPORT, 'm1/decisions.md': self.m1 / 'decisions.md',
                 'm1/tierB-queue.json': self.m1 / 'tierB-queue.json'}
        changed = {remote: local for remote, local in files.items() if digest(local) != self.state['uploaded'].get(remote)}
        if not changed:
            return []
        import re
        for remote, local in changed.items():
            if re.search(r'\bhf_[A-Za-z0-9]{30,}\b|\bghp_[A-Za-z0-9]{30,}\b', Path(local).read_text()):
                raise RuntimeError(f'Refusing to upload {remote}: it contains a token-shaped string')
        operations = [CommitOperationAdd(remote, str(local)) for remote, local in changed.items()]
        self.api.create_commit(REPO, repo_type='dataset', operations=operations,
                               commit_message=f'm1 {iso(utc_now())}: ' + ', '.join(sorted(changed)))
        for remote, local in changed.items():
            self.state['uploaded'][remote] = digest(local)
        return sorted(changed)

    def cycle(self):
        now = utc_now()
        self.decisions, self.escalations, self.first_frame_diffs = [], [], []
        download(self.api, self.inbox)
        wave1, wave2 = load(self.wave1_path), load(self.wave2_path)
        self.wave1 = wave1
        m2, lab = read_json(self.inbox / 'm2/status.json'), read_json(self.inbox / 'lab/status.json')
        copied = copy_m2_tier_a(wave2, self.inbox)
        delta = noise_floor(wave1)
        health = {'M2': self.health('M2', m2, now), 'lab': self.health('lab', lab, now)}
        calibration_text = self.calibration(read_json(self.inbox / 'm2/calibration.json'))
        kill_rows = self.kill_rule(wave1, wave2, status_runs(m2), delta)
        trust = self.lab_trust(read_json(self.inbox / 'lab/instances.json'), read_json(self.inbox / 'lab/env.json'))
        verdict, summaries = self.verdict(read_json(self.m1 / 'tierB-queue.json'), trust[0], wave1)
        self.render_report(self.coord_section(now, m2, lab, health, calibration_text, kill_rows, trust, verdict,
                                              summaries, wave2, copied, delta))
        uploaded = self.upload()
        self.save()
        print(json.dumps({'event': 'cycle', 'at': iso(now), 'copied': len(copied), 'decisions': self.decisions,
                          'uploaded': uploaded, 'm2': health['M2'][0], 'lab': health['lab'][0]}), flush=True)

    def save(self):
        temporary = self.state_path.with_name(self.state_path.name + '.tmp')
        temporary.write_text(json.dumps(self.state, indent=1))
        temporary.replace(self.state_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--wave1', type=Path, default=CHECKOUT / 'waves/wave1.json')
    parser.add_argument('--wave2', type=Path, default=CHECKOUT / 'waves/wave2.json')
    parser.add_argument('--interval', type=float, default=1800)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    from huggingface_hub import HfApi
    collector = Collector(HfApi(token=os.environ.get('HF_TOKEN')), args.wave1, args.wave2)
    while True:
        try:
            collector.cycle()
        except Exception as exc:  # keep collecting; the next cycle retries
            print(json.dumps({'event': 'cycle_failed', 'error': f'{type(exc).__name__}: {exc}'[:500]}), flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == '__main__':
    sys.exit(main())
