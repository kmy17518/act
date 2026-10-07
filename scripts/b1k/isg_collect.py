#!/usr/bin/env python3
"""Machine 1's collector for the ISG goal study's helper machines (plan §7, coordination repo).

Every cycle (default every 30 minutes):
  1. download `m2/` and `lab/` of the coordination repo into /tmp/dev/coord/in/ (a fresh copy each cycle);
  2. copy machine 2's Tier A results `m2/tierA-eval50/<run>/step_*.json` into `<runs_root>/<run>/tierA-eval50/` for runs
     any manifest (`--manifest`, repeatable) assigns to "M2" and that have no local training log;
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


def copy_m2_tier_a(manifests, inbox):
    """Copy machine 2's Tier A results into the local run directories of the runs any manifest assigns to M2."""
    copied = []
    for manifest in manifests:
        for name, run in manifest['_runs'].items():
            source = inbox / 'm2/tierA-eval50' / name
            run_dir = Path(manifest['runs_root']) / name
            if run.get('machine') != 'M2' or not source.is_dir():
                continue
            if (run_dir / 'metrics.jsonl').exists():
                print(json.dumps({'event': 'skip_local_run', 'run': name}), flush=True)  # never touch a local run
                continue
            target = run_dir / tier_a_config(manifest)['dir']
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
    """delta of plan §5 on relocalization (the only task in scope from 2026-10-07): max |seed 0 - seed 1| of the
    late tag-zero baseline's relocalization L1_own and gap on the test set over 40k and 50k."""
    deltas = {}
    for key in ('L1_own', 'gap'):
        values = []
        for step in (40000, 50000):
            a, b = (((tier_a(wave1, f'w1-late-tagzero-lrg1e-4-s{seed}', step) or {}).get('heldout', {}).get('tasks', {})
                     .get(GOAL_TASKS[0]) or {}).get(key) for seed in (0, 1))
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


RELOC = GOAL_TASKS[0]
SPLIT_ORDER = {'replay': 0, 'train': 1, 'eval50': 2}
SPLIT_SIZES = {'train': 100, 'eval50': 50}
FLOOR = 0.10  # plan §5 floor guard: no success-based comparison while every candidate is below ~10 % success
CORRECTION = ('Correction (2026-10-07): the evaluator does not need the policy to hold still. It stops the policy at the first '
              'step the camera pose is within tolerance, then holds the robot itself with no-op actions for max(50 steps, 20 % '
              'of the steps), adds a second window twice as long if the goal stops holding or something moved, and only then '
              'gives its verdict; `settle_windows: [10]` is only the short no-op window after a never-reached goal. The '
              'dataset\'s settle windows are this same epilogue, so `--settle-steps 0` matches the evaluator and the settle-20 '
              'rationale is withdrawn: r1-early-copy05-st20-s0 was stopped at step 975 (kept, marked stopped); GPU 2 trains '
              'r1-late-tagzero-st0-s1, GPU 0 queues r1-early-wrist-st0-s1, machine 2\'s GPU 1 runs r1-early-zero-st0-s0. Evidence: lab/env.json `episode_rules`; settling is not the '
              'bottleneck either (none of the 150 relocalization episodes of the three finalists was within tolerance at any step).')
METRICS = ('geodesic_distance_m', 'position_error_m', 'orientation_error_deg', 'within_tolerance')


def entry_dir(inbox, entry):
    if entry['split'] == 'replay':
        return inbox / 'lab/diagnostics/replay' / entry.get('replay_split', 'eval50')
    prefix = 'lab/tierB' if entry['split'] == 'eval50' else 'lab/tierB-train'
    return inbox / prefix / entry['run'] / f'step_{int(entry["step"]):08d}'


def entry_key(entry):
    return (entry['run'], entry.get('step'), entry['split'], entry.get('replay_split'))


def score_value(record, key):
    """A metric at the top level of an episode record or under its `score` dict (booleans count as 0/1)."""
    for source in (record, record.get('score') if isinstance(record.get('score'), dict) else {}):
        value = source.get(key)
        if isinstance(value, (bool, int, float)):
            return float(value)
    return None


def entry_results(inbox, entry):
    """({episode_index: record}, expected episode count, summary.json) of one queue entry (relocalization only)."""
    directory = entry_dir(inbox, entry)
    files = directory.glob('episode_*.json') if entry['split'] == 'replay' else (directory / RELOC).glob('episode_*.json')
    results = {}
    for path in files:
        try:
            results[int(path.stem.split('_')[1])] = read_json(path) or {}
        except (IndexError, ValueError):
            continue
    wanted = entry.get('episodes')
    if wanted is not None:
        results = {e: r for e, r in results.items() if e in set(wanted)}
    expected = len(wanted) if wanted is not None else SPLIT_SIZES.get(entry['split'], 0)
    return results, expected, read_json(directory / 'summary.json')


def summarize(results):
    valid = {e: r for e, r in results.items() if not r.get('error') and isinstance(r.get('success'), (bool, int, float))}
    row = {'done': len(results), 'valid': len(valid), 'errors': len(results) - len(valid),
           'success': sum(float(r['success']) for r in valid.values()) / len(valid) if valid else None}
    for key in METRICS:
        values = [v for v in (score_value(r, key) for r in valid.values()) if v is not None]
        row[key] = sum(values) / len(values) if values else None
    return row, valid


def paired(a, b, key, rng_seed=0, resamples=10000):
    """Mean of a - b over the common valid episodes (success or a continuous score), bootstrap 95 % CI."""
    values = [(float(a[e]['success']), float(b[e]['success'])) if key == 'success' else (score_value(a[e], key), score_value(b[e], key))
              for e in sorted(set(a) & set(b))]
    diff = np.array([x - y for x, y in values if x is not None and y is not None])
    if len(diff) < 2:
        return None
    means = diff[np.random.default_rng(rng_seed).integers(0, len(diff), (resamples, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'n': len(diff), 'diff': float(diff.mean()), 'ci': [float(lo), float(hi)]}


def reloc(result, key):
    return ((result or {}).get('heldout', {}).get('tasks', {}).get(RELOC) or {}).get(key)


class Collector:
    def __init__(self, api, manifests):
        self.api = api
        self.manifest_paths = [Path(p) for p in manifests]
        self.inbox, self.m1 = COORD / 'in', COORD / 'm1'
        self.state_path = COORD / 'state.json'
        self.state = read_json(self.state_path) or {'recorded': [], 'uploaded': {}, 'lab_trusted': False}
        self.decisions, self.escalations, self.first_frame_diffs = [], [], []

    def decide(self, key, audience, lines, log_line):
        """Append a decision once (keyed) to m1/decisions.md and the plan's decision log."""
        if key in self.state['recorded']:
            return False
        stamp = iso(utc_now())
        with (self.m1 / 'decisions.md').open('a') as stream:
            stream.write(f'\n## {stamp} — to {audience}\n' + ''.join(f'- {line}\n' for line in lines))
        plan = PLAN.read_text()
        with PLAN.open('a') as stream:
            stream.write(('' if plan.endswith('\n') else '\n') +
                         f'| {stamp[:10]} | coordination | {log_line} | `m1/decisions.md` entry {stamp} in `{REPO}` | see entry |\n')
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
            proposal = (' Recovery (plan §7, pre-approved by the user): machine 1 takes over its unfinished runs from their '
                        'Hub resume checkpoints under the same W&B ids as soon as a machine-1 GPU is free. M2 must not '
                        'restart any run before reading m1/decisions.md.' if name == 'M2' else '')
            self.decide(f'down:{name}:{iso(updated)}', 'all', [f'{name} is flagged as down: its status.json was last updated '
                        f'at {iso(updated)}, more than an hour ago.{proposal}'], f'{name} flagged as down (status.json older than 1 h)')
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

    def kill_rule(self, manifests, runs, delta):
        """Plan §5 triage at 20k for machine-2 runs, on relocalization numbers, against the manifest's base run."""
        rows = []
        for manifest in manifests:
            base_name = (manifest.get('base') or {}).get('runs', [None])[0]
            if not base_name:
                continue
            base_manifest = next((m for m in manifests if base_name in m['_runs']), manifest)
            base = tier_a(base_manifest, base_name, 20000)
            base_train, base_s = local_training(Path(manifest['runs_root']) / base_name, 19501, 20000)
            for name, run in manifest['_runs'].items():
                if run.get('machine') != 'M2' or name.startswith('m2-calib') or reloc(base, 'L1_own') is None:
                    continue
                result = tier_a(manifest, name, 20000)
                if result is None:
                    continue
                l1, gap, record = reloc(result, 'L1_own'), reloc(result, 'gap'), runs.get(name, {})
                train_l1, s_step = record.get('train_l1_last500'), record.get('s_per_step')
                reasons = []
                if isinstance(train_l1, (int, float)) and (train_l1 != train_l1 or (base_train and train_l1 > 2 * base_train)):
                    reasons.append(f'training L1 {train_l1:.4f} non-finite or > 2x the base run\'s {base_train:.4f}')
                if l1 is not None and gap is not None and delta['gap'] is not None and \
                        l1 >= 1.10 * reloc(base, 'L1_own') and gap <= delta['gap']:
                    reasons.append(f'relocalization L1_own {l1:.4f} >= 1.10 x base {reloc(base, "L1_own"):.4f} and gap '
                                   f'{gap:.4f} <= delta {delta["gap"]:.4f}')
                if isinstance(s_step, (int, float)) and base_s and s_step > 2 * base_s:
                    reasons.append(f'{s_step:.3f} s/step is below half the expected speed ({base_s:.3f} s/step)')
                text = (f'`{name}` at 20k: **kill** (plan §5): ' + '; '.join(reasons) + '. Stop it; its GPU waits for machine 1.'
                        if reasons else f'`{name}` at 20k: keep. Relocalization L1_own {fmt(l1)} vs base `{base_name}` '
                        f'{fmt(reloc(base, "L1_own"))}, gap {fmt(gap)} (delta {fmt(delta["gap"])}), {fmt(s_step, 3)} s/step.')
                self.decide(f'kill20k:{name}', 'M2', [text], f'M2 20k triage {name}: {"kill" if reasons else "keep"}')
                rows.append(text)
        return rows

    def lab_trust(self, instances, env):
        covered, parsed, diffs = instance_coverage(instances) if instances is not None else (0, 0, [])
        self.first_frame_diffs = diffs
        horizon = fixed_action_horizon(env)
        trusted = covered == 200 and horizon is not None
        reasons = [] if covered == 200 else [f'instances.json verifies {covered}/200 test episodes' if instances is not None
                                             else 'no instances.json yet']
        if horizon is None:
            reasons.append('env.json shows no single fixed action horizon' if env is not None else 'no env.json yet')
        if trusted != self.state.get('lab_trusted', False):
            self.state['lab_trusted'] = trusted
            self.decide(f'labtrust:{trusted}:{iso(utc_now())}', 'lab',
                        [f'Lab results are trusted from now on (instances 200/200, action horizon {horizon}).' if trusted else
                         'Lab results are no longer trusted: ' + '; '.join(reasons) + '.'], f'lab results {"trusted" if trusted else "not trusted"}')
        return trusted, horizon, covered, reasons

    # ------------------------------------------------------------------------------------------ Tier B queue

    def extend_queue(self, manifests, queue):
        """Append the manifests' planned evaluations (run field "tierB") whose eval checkpoint is on the Hub."""
        picks = read_json(CHECKOUT / 'splits/reloc-picks.json') or {}
        keys = {entry_key(e) for e in queue['entries']}
        added = []
        for manifest in manifests:
            for name, run in manifest['_runs'].items():
                for plan in run.get('tierB', []):
                    entry = {'run': name, 'step': int(plan['step']), 'split': plan['split']}
                    if entry_key(entry) in keys:
                        continue
                    repo, file = f'{manifest["hf_owner"]}/{manifest["hf_repo_prefix"]}{name}', f'eval/step-{int(plan["step"]):08d}.pt'
                    try:
                        info = self.api.get_paths_info(repo, [file], expand=True)
                    except Exception:  # repo not created yet
                        continue
                    if not info or not getattr(info[0], 'lfs', None):
                        continue
                    episodes = plan.get('episodes')
                    entry.update({'repo': repo, 'file': file, 'sha256': info[0].lfs.sha256, 'tasks': [RELOC],
                                  'episodes': picks.get(episodes) if isinstance(episodes, str) else episodes,
                                  'note': plan.get('note', f'{name} at {int(plan["step"]) // 1000}k on {plan["split"]}')})
                    queue['entries'].append(entry)
                    keys.add(entry_key(entry))
                    added.append(f'{name}@{plan["step"]}/{plan["split"]}')
        return added

    def funnel(self, queue):
        """Results per queue entry, the replay positive control and the eval50 step of the funnel."""
        rows, valid = [], {}
        for entry in queue['entries']:
            results, expected, summary = entry_results(self.inbox, entry)
            row, ok = summarize(results)
            row.update(entry=entry, expected=expected, complete=row['done'] >= expected > 0 and summary is not None)
            rows.append(row)
            valid[entry_key(entry)] = ok
        replays = [r for r in rows if r['entry']['split'] == 'replay']
        if replays and all(r['complete'] for r in replays):
            within = sum(int(bool(score_value(rec, 'within_tolerance') or rec.get('success')))
                         for r in replays for rec in valid[entry_key(r['entry'])].values())
            total = sum(r['valid'] for r in replays)
            passed = total > 0 and within == total and sum(r['errors'] for r in replays) == 0
            token = hashlib.sha256(f'{within}/{total}'.encode()).hexdigest()[:8]
            if self.decide(f'replay:{token}', 'lab', [
                    f'Positive control (open-loop replay of recorded actions): {within}/{total} episodes end within tolerance. '
                    + ('Passed: the action path is sound; continue with the queue.' if passed else
                       'Failed: the action path (base velocity semantics, control rate, settle check) is wrong; the queue is '
                       'paused until a re-run of the replays ends within tolerance for every episode.')],
                    f'replay positive control {"passed" if passed else "failed"} ({within}/{total})'):
                queue['paused'] = not passed
        for row in rows:
            entry = row['entry']
            if entry['split'] == 'train' and row['complete'] and (row['success'] or 0) > 0:
                target = dict(entry, split='eval50', episodes=None, note=f'{entry["run"]} at {entry["step"] // 1000}k: '
                              'succeeds on training instances, so eval50 (funnel step 3)')
                if entry_key(target) not in {entry_key(e) for e in queue['entries']}:
                    queue['entries'].append(target)
                self.decide(f'trainsuccess:{entry["run"]}:{entry["step"]}', 'all', [
                    f'`{entry["run"]}` at step {entry["step"]} succeeds on training instances ({row["success"]:.0%} of '
                    f'{row["valid"]}); its eval50 evaluation is queued.'], f'{entry["run"]}@{entry["step"]} succeeds on training instances')
        return rows, valid

    def comparisons(self, rows, valid):
        """Paired comparisons of complete entries on the same split and episodes, under the floor guard."""
        out, groups = [], {}
        for row in rows:
            entry = row['entry']
            if row['complete'] and entry['split'] != 'replay':
                groups.setdefault((entry['split'], tuple(entry.get('episodes') or ())), []).append(row)
        for (split, _), members in groups.items():
            if len(members) < 2:
                continue
            best = max((m['success'] or 0) for m in members)
            keys = ('success',) if best >= FLOOR else ('geodesic_distance_m', 'orientation_error_deg')
            out.append(f'- {split}: ' + ('success-based (some candidate reaches 10 %).' if best >= FLOOR else
                       f'floor guard: every candidate is below 10 % success (best {best:.0%}), so success is not compared; '
                       'paired final geodesic distance and orientation error (lower is better) instead.'))
            for a, b in combinations(members, 2):
                for key in keys:
                    result = paired(valid[entry_key(a['entry'])], valid[entry_key(b['entry'])], key)
                    if result:
                        out.append(f'  - `{a["entry"]["run"]}`@{a["entry"]["step"]} − `{b["entry"]["run"]}`@{b["entry"]["step"]}, '
                                   f'{key}: {result["diff"]:+.3f} (95 % CI [{result["ci"][0]:+.3f}, {result["ci"][1]:+.3f}], '
                                   f'n {result["n"]})')
        return out

    # ------------------------------------------------------------------------------------------ report

    def coord_section(self, now, m2, lab, health, calibration_text, kill_rows, trust, queue, rows, comparisons, copied):
        trusted, horizon, covered, reasons = trust
        asks = []
        for name, status in (('M2', m2), ('lab', lab)):
            for key in ('needs_decision', 'blocked'):
                items = (status or {}).get(key)
                asks += [f'- **{name} {key.replace("_", " ")}:** {item}' for item in (items if isinstance(items, list) else [items] if items else [])]
        asks += [f'- **machine 1 review:** {item}' for item in self.escalations]
        out = ['<!-- coord:begin -->', f'## Relocalization phase (collector, {now.astimezone(PDT):%Y-%m-%d %H:%M} PDT)', '',
               'Scope from 2026-10-07: camera_relocalization-standard only. Funnel: (1) open-loop replay of recorded actions '
               '(positive control) → (2) training instances → (3) eval50, only for checkpoints that succeed on training '
               f'instances plus each run\'s final checkpoint. Horizon {queue.get("horizon_steps")} steps; queue paused: '
               f'{queue.get("paused")}.', '', CORRECTION, '', '**Needs a decision / blocked:**', *(asks or ['- none']), '',
               '| Stage | Run | Step | Episodes | Success | Within tol. | Geodesic m | Position m | Orientation deg | Errors |',
               '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in sorted(rows, key=lambda r: SPLIT_ORDER.get(r['entry']['split'], 9)):
            entry = row['entry']
            stage = 'replay ' + entry.get('replay_split', '') if entry['split'] == 'replay' else entry['split']
            out.append(f'| {stage} | `{entry["run"]}` | {entry.get("step") or "—"} | {row["done"]}/{row["expected"]}'
                       f'{"" if row["complete"] else " (running)"} | {fmt(row["success"], 2)} | {fmt(row["within_tolerance"], 2)} | '
                       f'{fmt(row["geodesic_distance_m"], 2)} | {fmt(row["position_error_m"], 2)} | '
                       f'{fmt(row["orientation_error_deg"], 1)} | {row["errors"]} |')
        out += ['', '**Paired comparisons:**', *(comparisons or ['- none yet (needs two complete entries on the same episodes)'])]
        out += ['', f'**Helper health:** M2 — {health["M2"][0]}; lab — {health["lab"][0]}. Lab trust: '
                f'{"trusted" if trusted else "not trusted: " + "; ".join(reasons)} (instances {covered}/200, action horizon {horizon}).']
        runs = status_runs(m2)
        if runs:
            out += ['', '| M2 run | State | Step / max | s/step | Train L1 (last 500) | Peak GiB | ETA h | Exit |',
                    '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
            out += [f'| `{n}` | {r.get("state", "—")} | {r.get("step", "—")} / {r.get("max_steps", "—")} | {fmt(r.get("s_per_step"), 3)} | '
                    f'{fmt(r.get("train_l1_last500"))} | {fmt(r.get("peak_gib"), 0)} | {fmt(r.get("eta_h"), 1)} | {r.get("exit_code", "—")} |'
                    for n, r in runs.items() if r.get('state') not in ('finished',) or n.startswith('r1-')]
        out += ['', f'M2 calibration: {calibration_text}.', *[f'- {r}' for r in kill_rows],
                '', f'Copied this cycle: {len(copied)} machine-2 Tier A files. Decisions this cycle: '
                + ('; '.join(self.decisions) if self.decisions else 'none') + '.', '<!-- coord:end -->']
        return '\n'.join(out)

    def render_report(self, coord_text, manifests):
        from isg_report import BEGIN, END, render
        text = REPORT.read_text()
        sections = [(coord_text, '<!-- coord:begin -->', '<!-- coord:end -->', 'top')]
        for manifest, path in zip(manifests, self.manifest_paths):
            begin, end = (BEGIN, END) if path.stem == 'wave1' else (f'<!-- isg-runs-{path.stem}:begin -->', f'<!-- isg-runs-{path.stem}:end -->')
            try:
                body = render(manifest).replace(BEGIN, begin).replace(END, end)
                if begin != BEGIN:
                    body = body.replace(begin, begin + f'\n## {path.stem} tables (machine 1; machine-2 Tier A copied)', 1)
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
        import re
        from huggingface_hub import CommitOperationAdd
        files = {'docs/plan.md': PLAN, 'docs/report.md': REPORT, 'm1/decisions.md': self.m1 / 'decisions.md',
                 'm1/tierB-queue.json': self.m1 / 'tierB-queue.json'}
        changed = {remote: local for remote, local in files.items() if digest(local) != self.state['uploaded'].get(remote)}
        if not changed:
            return []
        for remote, local in changed.items():
            if re.search(r'\bhf_[A-Za-z0-9]{30,}\b|\bghp_[A-Za-z0-9]{30,}\b', Path(local).read_text()):
                raise RuntimeError(f'Refusing to upload {remote}: it contains a token-shaped string')
        self.api.create_commit(REPO, repo_type='dataset', operations=[CommitOperationAdd(r, str(l)) for r, l in changed.items()],
                               commit_message=f'm1 {iso(utc_now())}: ' + ', '.join(sorted(changed)))
        for remote, local in changed.items():
            self.state['uploaded'][remote] = digest(local)
        return sorted(changed)

    def cycle(self):
        now = utc_now()
        self.decisions, self.escalations, self.first_frame_diffs = [], [], []
        download(self.api, self.inbox)
        manifests = [load(p) for p in self.manifest_paths]
        m2, lab = read_json(self.inbox / 'm2/status.json'), read_json(self.inbox / 'lab/status.json')
        copied = copy_m2_tier_a(manifests, self.inbox)
        delta = noise_floor(manifests[0])
        health = {'M2': self.health('M2', m2, now), 'lab': self.health('lab', lab, now)}
        calibration_text = self.calibration(read_json(self.inbox / 'm2/calibration.json'))
        kill_rows = self.kill_rule(manifests, status_runs(m2), delta)
        trust = self.lab_trust(read_json(self.inbox / 'lab/instances.json'), read_json(self.inbox / 'lab/env.json'))
        queue_path = self.m1 / 'tierB-queue.json'
        queue = read_json(queue_path)
        before = json.dumps(queue, sort_keys=True)
        added = self.extend_queue(manifests, queue)
        rows, valid = self.funnel(queue)
        queue['entries'].sort(key=lambda e: SPLIT_ORDER.get(e['split'], 9))
        if json.dumps(queue, sort_keys=True) != before:
            queue['updated_at'] = iso(now)
            queue_path.write_text(json.dumps(queue, indent=2) + '\n')
        self.render_report(self.coord_section(now, m2, lab, health, calibration_text, kill_rows, trust, queue, rows,
                                              self.comparisons(rows, valid), copied), manifests)
        uploaded = self.upload()
        self.save()
        print(json.dumps({'event': 'cycle', 'at': iso(now), 'copied': len(copied), 'queued': added, 'decisions': self.decisions,
                          'uploaded': uploaded, 'm2': health['M2'][0], 'lab': health['lab'][0]}), flush=True)

    def save(self):
        temporary = self.state_path.with_name(self.state_path.name + '.tmp')
        temporary.write_text(json.dumps(self.state, indent=1))
        temporary.replace(self.state_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, action='append',
                        help='Wave manifests in report order (repeat); default waves/wave1.json, wave2.json, reloc1.json')
    parser.add_argument('--interval', type=float, default=1800)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    from huggingface_hub import HfApi
    manifests = args.manifest or [CHECKOUT / f'waves/{n}.json' for n in ('wave1', 'wave2', 'reloc1')]
    collector = Collector(HfApi(token=os.environ.get('HF_TOKEN')), manifests)
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
