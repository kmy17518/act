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
  6. keep the lab queue `m1/tierB-queue.json` (isg-tierB-queue/v2): append the manifests' planned evaluations, queue
     eval50 for checkpoints that succeed on training episodes, order it (replays, execution-setting tests, training
     instances, eval50) and estimate the lab hours it holds. Entries may carry `tag` and `server_args` (extra
     serve_b1k.py arguments); tagged results live in lab/diagnostics/exec/<tag>/<run>/step_XXXXXXXX/, and every episode
     must report the setting it was queued with. Once the execution-setting test is complete (with the lab's own tests
     of `exec_test.lab_runs` in the manifests) the eval50 setting is chosen once and applied to unstarted eval50 entries;
  7. check that this checkout reached GitHub (`git ls-remote` of kmy17518/act isg-wave2-dev, the branch machine 2 and
     the lab pull; this checkout's `origin` is a local clone): unpushed commits or uncommitted manifest changes are
     reported at the top of docs/report.md, and a decision naming a commit or runs that GitHub lacks is held;
  8. upload `docs/` and `m1/` in one commit when anything changed.
It never writes under `m2/` or `lab/`, never starts uploaders, never pushes, and writes locally only under /tmp.

    isg_collect.py [--interval 1800] [--once]
    isg_collect.py --decide M2|lab|all --line TEXT [--line TEXT ...] [--log TEXT]   (manual decision, same push check)
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
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
SPEC = Path('/tmp/dev/docs/isg-correction-data-spec.md')
WRIST_BEGIN, WRIST_END = '<!-- wrist-control:begin -->', '<!-- wrist-control:end -->'
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
    """The default action horizon of env.json: its top-level `action_horizon` (per-entry `server_args` may run other
    settings), else every value under a key containing 'action_horizon' (at any depth) must be the same integer."""
    top = (env or {}).get('action_horizon')
    if isinstance(top, int) and not isinstance(top, bool):
        return top
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
EXEC_DIR = 'lab/diagnostics/exec'  # results of queue entries with a "tag" (and of the lab's own execution tests)
DEFAULT_SETTING = 'ah16'  # serve_b1k.py's ACT default: chunked execution, 16 actions per query
TOL_POSITION_M, TOL_ORIENTATION_DEG = 0.05, math.degrees(0.1)
MIN_PAIRS = 10  # paired training episodes a setting needs before it can be chosen for eval50
LAB_HOURS_WARN = 24.0


def setting_args(setting):
    """serve_b1k.py arguments of an execution setting: 'ta' (temporal aggregation) or 'ah<N>' (N actions per query)."""
    if setting == 'ta':
        return ['--temporal-agg']
    return [] if setting == DEFAULT_SETTING else ['--action-horizon', setting[2:]]


def setting_of_args(args):
    """Execution setting of serve_b1k.py arguments: 'ta' with --temporal-agg, else 'ah<N>' (--action-horizon, default 16)."""
    args = [str(a) for a in (args or [])]
    if '--temporal-agg' in args:
        return 'ta'
    for i, arg in enumerate(args):
        if arg == '--action-horizon' and i + 1 < len(args):
            return f'ah{int(args[i + 1])}'
        if arg.startswith('--action-horizon='):
            return f'ah{int(arg.split("=", 1)[1])}'
    return DEFAULT_SETTING


def record_setting(record):
    """Setting an episode actually ran with: the server's own metadata, else its recorded command line (None if unknown)."""
    meta = record.get('server_metadata') if isinstance(record.get('server_metadata'), dict) else {}
    if meta.get('execution_mode') == 'temporal_aggregation' or meta.get('temporal_agg') is True:
        return 'ta'
    horizon = meta.get('action_horizon')
    if isinstance(horizon, int) and not isinstance(horizon, bool):
        return f'ah{horizon}'
    command = record.get('server_args')
    if isinstance(command, str):
        command = command.split()
    return setting_of_args(command) if isinstance(command, list) else None


GITHUB_URL, GITHUB_BRANCH = 'https://github.com/kmy17518/act.git', 'isg-wave2-dev'  # what machine 2 and the lab pull
GITHUB_REF = 'refs/remotes/github/isg-wave2-dev'  # last fetched copy; this checkout's `origin` is a local clone
RUN_NAME = re.compile(r'`((?:w1|w2|r1)-[a-z0-9.-]+-s\d+)`')
COMMIT_LIKE = re.compile(r'\b[0-9a-f]{7,40}\b')
PUSH_COMMAND = ('git -C /tmp/dev/baselines/act-isg-w2 push origin isg-wave2-dev && '
                'git -C /tmp/dev/baselines/act push origin isg-wave2-dev')


def git(*args, check=True):
    result = subprocess.run(['git', '-C', str(CHECKOUT), *args], capture_output=True, text=True, timeout=120)
    if check and result.returncode:
        raise RuntimeError(f'git {" ".join(args[:2])}: {result.stderr.strip()[:200]}')
    return result


def github_state(manifest_paths):
    """This checkout against GitHub's isg-wave2-dev: `git ls-remote`, then a fetch into GITHUB_REF for ancestry and the
    pushed manifests' run names. When GitHub cannot be reached, the last fetched copy stands in (`ok` is False)."""
    error = None
    try:
        listed = git('ls-remote', GITHUB_URL, f'refs/heads/{GITHUB_BRANCH}').stdout.split()
        if not listed:
            raise RuntimeError(f'GitHub has no branch {GITHUB_BRANCH}')
        git('fetch', '--quiet', GITHUB_URL, f'+refs/heads/{GITHUB_BRANCH}:{GITHUB_REF}')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'[:300]
    if git('rev-parse', '--verify', '--quiet', GITHUB_REF, check=False).returncode:
        return {'ok': False, 'error': error or 'no fetched copy of the GitHub branch'}
    runs = set()
    for path in manifest_paths:
        path = Path(path)
        try:
            relative = path.resolve().relative_to(CHECKOUT) if path.is_absolute() else path
        except ValueError:
            continue
        shown = git('show', f'{GITHUB_REF}:{relative.as_posix()}', check=False)
        if shown.returncode == 0:
            manifest = json.loads(shown.stdout)
            runs |= {r['name'] for r in manifest.get('runs', []) + manifest.get('deferred', [])}
    head = git('rev-parse', 'HEAD').stdout.strip()
    return {'ok': error is None, 'error': error, 'remote': git('rev-parse', GITHUB_REF).stdout.strip(), 'head': head,
            'pushed': git('merge-base', '--is-ancestor', head, GITHUB_REF, check=False).returncode == 0,
            'unpushed': git('log', '--oneline', f'{GITHUB_REF}..HEAD').stdout.splitlines(),
            'dirty': git('status', '--porcelain', '--', 'waves', 'splits').stdout.splitlines(), 'runs': sorted(runs)}


def push_gate(text, github, local_runs):
    """Why a decision may not be posted yet: it names a commit of this checkout that GitHub's branch lacks, or a run of
    the local manifests that the manifests on GitHub lack (machine 2 starts runs from what it pulls)."""
    commits = [c for c in dict.fromkeys(COMMIT_LIKE.findall(text))
               if git('rev-parse', '--verify', '--quiet', f'{c}^{{commit}}', check=False).returncode == 0]
    runs = [r for r in dict.fromkeys(RUN_NAME.findall(text)) if r in local_runs]
    if not commits and not runs:
        return []
    if 'runs' not in github:
        return [f'GitHub could not be checked ({github.get("error")})']
    reasons = [f'commit {c} is not on GitHub' for c in commits
               if git('merge-base', '--is-ancestor', c, GITHUB_REF, check=False).returncode != 0]
    missing = [r for r in runs if r not in set(github['runs'])]
    if missing:
        reasons.append('runs ' + ', '.join(f'`{r}`' for r in missing) + ' are not in the manifests on GitHub')
    return reasons


def push_status(github):
    """Report line on whether machine 1's work reached GitHub, and the failure to escalate (None when pushed)."""
    if 'runs' not in github:
        problem = f'GitHub could not be checked: {github.get("error")}'
        return f'**GitHub push check:** {problem}.', problem
    stale = '' if github['ok'] else f' (last fetched copy; GitHub unreachable now: {github["error"]})'
    if github['pushed'] and not github['dirty']:
        return (f'**GitHub push check:** `{GITHUB_BRANCH}` on GitHub is at {github["remote"][:7]}, which contains machine 1\'s '
                f'HEAD {github["head"][:7]}{stale}.'), None
    parts = []
    if not github['pushed']:
        parts.append(f'machine 1\'s HEAD {github["head"][:7]} has {len(github["unpushed"])} commits that GitHub\'s '
                     f'`{GITHUB_BRANCH}` ({github["remote"][:7]}) lacks: ' + '; '.join(github['unpushed'][:5]))
    if github['dirty']:
        parts.append('uncommitted manifest or split changes: ' + ', '.join(line[3:] for line in github['dirty'][:5]))
    problem = 'NOT PUSHED: ' + '; '.join(parts) + f'{stale}. Push with `{PUSH_COMMAND}`'
    return f'**GitHub push check: {problem}.**', problem


def planned(queue, run, step, split):
    """Whether the queue already holds a planned evaluation: eval50 under any execution setting, training instances only
    untagged (tagged training entries are execution-setting tests)."""
    return any(e['run'] == run and e.get('step') == step and e['split'] == split and (split == 'eval50' or not e.get('tag'))
               for e in queue['entries'])


def queue_rank(entry):
    """Queue order by stage: replays, execution-setting tests (tagged training entries), training instances, eval50."""
    if entry['split'] == 'train':
        return 1 if entry.get('tag') else 2
    return {'replay': 0, 'eval50': 3}.get(entry['split'], 9)


def entry_dir(inbox, entry):
    if entry['split'] == 'replay':
        return inbox / 'lab/diagnostics/replay' / entry.get('replay_split', 'eval50')
    if entry.get('tag'):
        return inbox / EXEC_DIR / entry['tag'] / entry['run'] / f'step_{int(entry["step"]):08d}'
    prefix = 'lab/tierB' if entry['split'] == 'eval50' else 'lab/tierB-train'
    return inbox / prefix / entry['run'] / f'step_{int(entry["step"]):08d}'


def entry_key(entry):
    return (entry['run'], entry.get('step'), entry['split'], entry.get('replay_split'), entry.get('tag'))


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
    return results, expected, read_json(directory / 'summary.json') or read_json(directory / RELOC / 'summary.json')


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


def valid_record(record):
    return not record.get('error') and isinstance(record.get('success'), (bool, int, float))


def reached_tolerance(record):
    """Replay positive control: the replay reached tolerance at some step (stop-on-success fired or the goal held),
    whatever the evaluator's final no-op hold measured; boundary cases such as 0.050 m against 0.05 m pass."""
    first = record.get('first_success_step')
    return bool(record.get('success') or record.get('binary_success_ever') or score_value(record, 'within_tolerance')
                or (isinstance(first, (int, float)) and not isinstance(first, bool) and first >= 0))


def tolerance_ratio(record):
    """max(position / 0.05 m, orientation / 0.1 rad) at the final step: at most 1 means within tolerance."""
    position, orientation = score_value(record, 'position_error_m'), score_value(record, 'orientation_error_deg')
    if position is None or orientation is None:
        return None
    return max(position / TOL_POSITION_M, orientation / TOL_ORIENTATION_DEG)


def exec_results(inbox):
    """Execution-setting results on training episodes: ({(run, step): {setting: {episode: record}}}, {directory: complete})
    from every tagged result directory (queue entries and the lab's own tests), with the default results of
    lab/tierB-train for the same checkpoints. The setting is the one the server reported, not the tag."""
    results, complete = {}, {}
    for directory in sorted((inbox / EXEC_DIR).glob('*/*/step_*')):
        run, step = directory.parts[-2], int(directory.name[5:])
        files = sorted((directory / RELOC).glob('episode_*.json'))
        records = [(int(p.stem.split('_')[1]), read_json(p) or {}) for p in files]
        records = [(e, r) for e, r in records if r.get('split') in (None, 'train')]
        if not records:
            continue
        summary = read_json(directory / 'summary.json') or read_json(directory / RELOC / 'summary.json')
        complete[str(directory.relative_to(inbox))] = summary is not None and bool(summary.get('complete', True))
        for episode, record in records:
            setting = record_setting(record) or directory.parts[-3]
            results.setdefault((run, step), {}).setdefault(setting, {})[episode] = record
    for (run, step), by_setting in results.items():
        for path in (inbox / 'lab/tierB-train' / run / f'step_{step:08d}' / RELOC).glob('episode_*.json'):
            record = read_json(path) or {}
            if (record_setting(record) or DEFAULT_SETTING) == DEFAULT_SETTING:
                by_setting.setdefault(DEFAULT_SETTING, {})[int(path.stem.split('_')[1])] = record
    return results, complete


def exec_table(results):
    """Per setting against the default on the same (checkpoint, episode), pooled over checkpoints: paired episodes,
    successes of both, and the mean paired change in log tolerance ratio (negative: closer to the goal) with a bootstrap
    95 % CI."""
    table = {}
    for (run, step), by_setting in sorted(results.items()):
        base = by_setting.get(DEFAULT_SETTING, {})
        for setting, records in by_setting.items():
            if setting == DEFAULT_SETTING:
                continue
            row = table.setdefault(setting, {'diffs': [], 'success': 0, 'base_success': 0, 'checkpoints': []})
            for episode, record in sorted(records.items()):
                other = base.get(episode)
                if other is None or not valid_record(record) or not valid_record(other):
                    continue
                a, b = tolerance_ratio(record), tolerance_ratio(other)
                if a is None or b is None:
                    continue
                row['diffs'].append(math.log(max(a, 1e-6)) - math.log(max(b, 1e-6)))
                row['success'] += int(bool(record['success']))
                row['base_success'] += int(bool(other['success']))
                if f'{run}@{step // 1000}k' not in row['checkpoints']:
                    row['checkpoints'].append(f'{run}@{step // 1000}k')
    for row in table.values():
        diff = np.array(row.pop('diffs'))
        row['n'] = len(diff)
        if len(diff) >= 2:
            means = diff[np.random.default_rng(0).integers(0, len(diff), (10000, len(diff)))].mean(axis=1)
            row.update(mean=float(diff.mean()), ci=[float(x) for x in np.percentile(means, [2.5, 97.5])])
        else:
            row.update(mean=float(diff.mean()) if len(diff) else None, ci=None)
    return table


def choose_setting(table):
    """eval50 execution setting from the execution-test table, or None while no setting has MIN_PAIRS paired episodes:
    the most extra successes over the default if some setting gains at least 2; otherwise the lowest paired log tolerance
    ratio whose 95 % CI lies below 0 without losing successes; otherwise the default."""
    candidates = {s: t for s, t in table.items() if t['n'] >= MIN_PAIRS and t['ci'] is not None}
    if not candidates:
        return None, 'no setting has %d paired training episodes yet' % MIN_PAIRS
    gains = {s: t['success'] - t['base_success'] for s, t in candidates.items()}
    if max(gains.values()) >= 2:
        setting = max(candidates, key=lambda s: (gains[s], -candidates[s]['mean']))
        t = candidates[setting]
        return setting, (f'{t["success"]} successes against {t["base_success"]} for the default on the same {t["n"]} '
                         f'training episodes ({", ".join(t["checkpoints"])})')
    better = [s for s, t in candidates.items() if t['ci'][1] < 0 and gains[s] >= 0]
    if better:
        setting = min(better, key=lambda s: candidates[s]['mean'])
        t = candidates[setting]
        return setting, (f'final error (tolerance ratio) {math.exp(t["mean"]):.2f}x the default, 95 % CI '
                         f'[{math.exp(t["ci"][0]):.2f}, {math.exp(t["ci"][1]):.2f}], on {t["n"]} paired training episodes '
                         f'({", ".join(t["checkpoints"])}); successes {t["success"]} vs {t["base_success"]}')
    return DEFAULT_SETTING, ('no setting gains 2 successes or lowers the final error with a 95 % CI below 1x: ' +
                             '; '.join(f'{s} {math.exp(t["mean"]):.2f}x [{math.exp(t["ci"][0]):.2f}, {math.exp(t["ci"][1]):.2f}], '
                                       f'{t["success"]} vs {t["base_success"]} successes, n {t["n"]}'
                                       for s, t in sorted(candidates.items())))


def episode_minutes(inbox, env, horizon):
    """Lab minutes per episode as observed: the mean gap between consecutive finishes of the last 50 policy episodes at
    the queue's horizon (training instances, execution tests, eval50), ignoring gaps over 20 minutes (idle lab); else
    env.json's `minutes_per_episode` over its parallel evaluators."""
    records = [read_json(p) or {} for p in list((inbox / 'lab/tierB-train').glob(f'*/step_*/{RELOC}/episode_*.json'))
               + list((inbox / EXEC_DIR).glob(f'*/*/step_*/{RELOC}/episode_*.json'))
               + list((inbox / 'lab/tierB').glob(f'*/step_*/{RELOC}/episode_*.json'))]
    times = sorted(t for t in (parse_time(r.get('finished_at')) for r in records if r.get('horizon') == horizon)
                   if t is not None)[-50:]
    gaps = [(b - a).total_seconds() / 60 for a, b in zip(times, times[1:]) if 0 <= (b - a).total_seconds() <= 1200]
    if len(gaps) >= 10:
        return float(np.mean(gaps)), f'observed finish rate of the last {len(gaps) + 1} episodes'
    parallel = max(int((env or {}).get('parallel_evaluators') or 1), 1)
    minutes = (env or {}).get('minutes_per_episode')
    return (float(minutes) / parallel if minutes else 5.0 / parallel), f'env.json, {parallel} evaluators'


class Collector:
    def __init__(self, api, manifests):
        self.api = api
        self.manifest_paths = [Path(p) for p in manifests]
        self.inbox, self.m1 = COORD / 'in', COORD / 'm1'
        self.state_path = COORD / 'state.json'
        self.state = read_json(self.state_path) or {'recorded': [], 'uploaded': {}, 'lab_trusted': False}
        self.decisions, self.escalations, self.first_frame_diffs = [], [], []
        self.github, self.local_runs = {}, set()
        self.wrist_control_final = None

    def log_plan(self, key, area, decision, evidence, floor):
        """Append one row to the plan's decision log only (no m1/decisions.md entry), once per key."""
        if key in self.state['recorded']:
            return False
        plan = PLAN.read_text()
        with PLAN.open('a') as stream:
            stream.write(('' if plan.endswith('\n') else '\n') + f'| {iso(utc_now())[:10]} | {area} | {decision} | {evidence} | {floor} |\n')
        self.state['recorded'].append(key)
        self.decisions.append(f'plan log: {decision[:80]}')
        return True

    def decide(self, key, audience, lines, log_line):
        """Append a decision once (keyed) to m1/decisions.md and the plan's decision log, unless it names commits or runs
        that have not reached GitHub (then it is held and escalated; the next cycle retries)."""
        if key in self.state['recorded']:
            return False
        reasons = push_gate(' '.join(lines), self.github, self.local_runs)
        if reasons:
            self.escalations.append(f'decision to {audience} held until the push lands ({log_line}): ' + '; '.join(reasons))
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

    def eval50_setting(self, entry):
        """Give a new eval50 entry the execution setting chosen from the execution test (no-op before the choice)."""
        chosen = self.state.get('eval50_setting') or {}
        if entry['split'] == 'eval50' and chosen.get('tag'):
            entry.update(tag=chosen['tag'], server_args=list(chosen['server_args']))
        return entry

    def extend_queue(self, manifests, queue):
        """Append the manifests' planned evaluations (run field "tierB") whose eval checkpoint is on the Hub."""
        picks = read_json(CHECKOUT / 'splits/reloc-picks.json') or {}
        added = []
        for manifest in manifests:
            for name, run in manifest['_runs'].items():
                for plan in run.get('tierB', []):
                    entry = {'run': name, 'step': int(plan['step']), 'split': plan['split']}
                    if planned(queue, name, entry['step'], entry['split']):
                        continue
                    repo, file = f'{manifest["hf_owner"]}/{manifest["hf_repo_prefix"]}{name}', f'eval/step-{int(plan["step"]):08d}.pt'
                    try:
                        info = self.api.get_paths_info(repo, [file], expand=True)
                    except Exception:  # repo not created yet
                        continue
                    if not info or not getattr(info[0], 'lfs', None):
                        continue
                    episodes = plan.get('episodes')
                    if plan['split'] == 'eval50' and episodes is None and picks.get('eval25'):
                        episodes = 'eval25'  # stage 1; funnel() completes it to all 50 when warranted
                    entry.update({'repo': repo, 'file': file, 'sha256': info[0].lfs.sha256, 'tasks': [RELOC],
                                  'episodes': picks.get(episodes) if isinstance(episodes, str) else episodes,
                                  'note': plan.get('note', f'{name} at {int(plan["step"]) // 1000}k on {plan["split"]}')})
                    queue['entries'].append(self.eval50_setting(entry))
                    added.append(f'{name}@{plan["step"]}/{plan["split"]}')
        return added

    def funnel(self, queue):
        """Results per queue entry, the replay positive control and the eval50 step of the funnel."""
        rows, valid = [], {}
        for entry in queue['entries']:
            results, expected, summary = entry_results(self.inbox, entry)
            wanted = setting_of_args(entry.get('server_args'))
            wrong = sorted(e for e, r in results.items() if record_setting(r) not in (None, wanted))
            if wrong:
                self.escalations.append(
                    f'`{entry["run"]}`@{entry.get("step")} {entry["split"]}{" [" + entry["tag"] + "]" if entry.get("tag") else ""}: '
                    f'episodes {wrong} ran with server setting {sorted({record_setting(results[e]) for e in wrong})}, not '
                    f'{wanted} as queued (server_args {entry.get("server_args") or []}); they are left out until re-run.')
                results = {e: r for e, r in results.items() if e not in wrong}
            row, ok = summarize(results)
            row.update(entry=entry, expected=expected, complete=row['done'] >= expected > 0 and summary is not None,
                       episodes_done=sorted(results))
            rows.append(row)
            valid[entry_key(entry)] = ok
        replays = [r for r in rows if r['entry']['split'] == 'replay']
        if replays and all(r['complete'] for r in replays):
            within = sum(int(reached_tolerance(rec)) for r in replays for rec in valid[entry_key(r['entry'])].values())
            total = sum(r['valid'] for r in replays)
            passed = total > 0 and within == total and sum(r['errors'] for r in replays) == 0
            token = hashlib.sha256(f'{within}/{total}'.encode()).hexdigest()[:8]
            if self.decide(f'replay:{token}', 'lab', [
                    f'Positive control (open-loop replay of recorded actions): {within}/{total} episodes reach tolerance at '
                    'some step. ' + ('Passed: the action path is sound; continue with the queue.' if passed else
                                     'Failed: the action path (base velocity semantics, control rate, settle check) is wrong; '
                                     'the queue is paused until a re-run of the replays reaches tolerance in every episode.')],
                    f'replay positive control {"passed" if passed else "failed"} ({within}/{total})'):
                queue['paused'] = not passed
        for row in rows:
            entry = row['entry']
            if entry['split'] == 'train' and row['valid'] and (row['success'] or 0) > 0:
                setting = setting_of_args(entry.get('server_args'))
                target = {k: v for k, v in entry.items() if k not in ('tag', 'server_args')}
                target.update(split='eval50', episodes=None, note=f'{entry["run"]} at {entry["step"] // 1000}k: '
                              'succeeds on training instances, so eval50 (funnel step 3)')
                done = summarize(entry_results(self.inbox, target)[0])[0]
                if planned(queue, entry['run'], entry['step'], 'eval50'):
                    status = 'its eval50 evaluation is queued'
                elif done['valid'] >= SPLIT_SIZES['eval50']:
                    status = f'it already has a complete eval50 result ({done["success"]:.0%} success), so nothing is re-queued'
                else:
                    queue['entries'].append(self.eval50_setting(target))
                    status = 'its eval50 evaluation is queued'
                self.decide(f'trainsuccess:{entry["run"]}:{entry["step"]}', 'lab', [
                    f'`{entry["run"]}` at step {entry["step"]} succeeds on training instances '
                    f'({round(row["success"] * row["valid"])} of {row["valid"]} episodes so far, execution setting {setting}); '
                    f'{status}.'], f'{entry["run"]}@{entry["step"]} succeeds on training instances')
        self.complete_eval50(queue, rows)
        return rows, valid

    def complete_eval50(self, queue, rows):
        """Two-stage eval50: an entry listing the 25 stage-1 episodes (splits/reloc-picks.json `eval25`) becomes the full
        eval50 set when any of them succeeds or its checkpoint succeeds on training episodes (any execution setting)."""
        stage1 = sorted((read_json(CHECKOUT / 'splits/reloc-picks.json') or {}).get('eval25') or [])
        trained = {(r['entry']['run'], r['entry']['step']) for r in rows
                   if r['entry']['split'] == 'train' and r['valid'] and (r['success'] or 0) > 0}
        for row in rows:
            entry = row['entry']
            if entry['split'] != 'eval50' or not stage1 or sorted(entry.get('episodes') or []) != stage1:
                continue
            successes = round((row['success'] or 0) * row['valid'])
            if successes or (entry['run'], entry['step']) in trained:
                entry['episodes'] = None
                why = (f'{successes} of its first {row["valid"]} stage-1 episodes succeed' if successes else
                       'its checkpoint succeeds on training episodes')
                self.decide(f'eval50full:{entry["run"]}:{entry["step"]}:{entry.get("tag")}', 'lab', [
                    f'`{entry["run"]}`@{entry["step"] // 1000}k eval50: {why}, so the entry now covers all 50 test episodes '
                    '(`"episodes": null`). Run only the missing ones and keep the episode files you have.'],
                    f'{entry["run"]}@{entry["step"]} eval50 completed to 50')

    def exec_decision(self, queue, rows, manifests):
        """Execution-setting test: the per-setting table for the report and, once every test result is in (queued
        entries and the lab's own tests listed under the manifests' `exec_test.lab_runs`), the eval50 setting (decided
        once, applied to the eval50 entries that have not started)."""
        results, complete = exec_results(self.inbox)
        table = exec_table(results)
        queued = [r for r in rows if r['entry']['split'] == 'train' and r['entry'].get('tag')]
        waiting = [f'`{r["entry"]["run"]}`@{r["entry"]["step"] // 1000}k {r["entry"]["tag"]} ({r["done"]}/{r["expected"]})'
                   for r in queued if not r['complete']]
        waiting += [f'{path} (no complete summary.json)' for path, done in complete.items() if not done]
        for manifest in manifests:
            for test in (manifest.get('exec_test') or {}).get('lab_runs', []):
                have = results.get((test['run'], int(test['step'])), {})
                waiting += [f'lab test `{test["run"]}`@{int(test["step"]) // 1000}k {s}' for s in test['settings'] if s not in have]
        chosen = self.state.get('eval50_setting')
        if chosen is None and not waiting and table:
            setting, reason = choose_setting(table)
            if setting is not None:
                chosen = {'setting': setting, 'server_args': setting_args(setting),
                          'tag': None if setting == DEFAULT_SETTING else f'{setting}-e50', 'decided_at': iso(utc_now()), 'reason': reason}
                self.state['eval50_setting'] = chosen
                kept = []
                for row in rows:
                    entry = row['entry']
                    if entry['split'] != 'eval50' or entry.get('tag'):
                        continue
                    if row['done']:
                        kept.append(f'`{entry["run"]}`@{entry["step"] // 1000}k ({row["done"]} episodes done)')
                    else:
                        self.eval50_setting(entry)
                self.decide(f'eval50setting:{setting}', 'lab', [
                    f'eval50 runs with execution setting **{setting}** (serve_b1k.py {" ".join(chosen["server_args"]) or "defaults"}): '
                    f'{reason}.',
                    (f'Every eval50 entry that has not started now carries `"tag": "{chosen["tag"]}"` and `"server_args": '
                     f'{json.dumps(chosen["server_args"])}`; its results go to {EXEC_DIR}/{chosen["tag"]}/<run>/step_XXXXXXXX/. '
                     if chosen['tag'] else 'eval50 entries are unchanged. ') +
                    (f'Entries already started keep the default: {", ".join(kept)}.' if kept else '')],
                    f'eval50 execution setting: {setting}')
        return table, waiting, chosen

    def lab_load(self, queue, rows, manifests, env):
        """Lab episodes left in the queue and projected from the manifests' planned evaluations not queued yet, in hours."""
        instances = (read_json(self.inbox / 'lab/instances-train.json') or {}).get('episodes') or {}
        mapped = {int(e) for e, v in instances.items() if isinstance(v, dict) and v.get('verified')}
        picks = read_json(CHECKOUT / 'splits/reloc-picks.json') or {}

        def runnable(test_set, episodes, done=()):
            """Episodes the lab can still run: every test-set episode is verified; training episodes only when mapped."""
            if episodes is None:
                return max(SPLIT_SIZES['eval50'] - len(done), 0)
            todo = set(episodes) - set(done)
            return len(todo) if test_set else len(todo & mapped)

        queued = sum(runnable(row['entry']['split'] == 'eval50' or row['entry'].get('replay_split') == 'eval50',
                              row['entry'].get('episodes'), row['episodes_done']) for row in rows)
        projected = 0
        for manifest in manifests:
            for name, run in manifest['_runs'].items():
                if run.get('stopped'):
                    continue
                for plan in run.get('tierB', []):
                    if not planned(queue, name, int(plan['step']), plan['split']):
                        episodes = plan.get('episodes')
                        if plan['split'] == 'eval50' and episodes is None and picks.get('eval25'):
                            episodes = 'eval25'
                        projected += runnable(plan['split'] == 'eval50', picks.get(episodes) if isinstance(episodes, str) else episodes)
        minutes, source = episode_minutes(self.inbox, env, queue.get('horizon_steps'))
        return {'queued': queued, 'projected': projected, 'hours_queued': queued * minutes / 60,
                'hours_projected': projected * minutes / 60, 'minutes': minutes, 'source': source, 'mapped': len(mapped)}

    def comparisons(self, rows, valid):
        """Paired comparisons of complete entries on the same split and episodes, under the floor guard."""
        out, groups = [], {}
        for row in rows:
            entry = row['entry']
            if row['complete'] and entry['split'] != 'replay':
                groups.setdefault((entry['split'], tuple(entry.get('episodes') or ()), entry.get('tag')), []).append(row)
        for (split, _, tag), members in groups.items():
            if len(members) < 2:
                continue
            best = max((m['success'] or 0) for m in members)
            keys = ('success',) if best >= FLOOR else ('geodesic_distance_m', 'orientation_error_deg')
            out.append(f'- {split}{" [" + tag + "]" if tag else ""}: ' + ('success-based (some candidate reaches 10 %).' if best >= FLOOR else
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

    def coord_section(self, now, m2, lab, health, calibration_text, kill_rows, trust, queue, rows, comparisons, copied,
                      execution, load, push_line):
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
               'instances plus each run\'s final checkpoint, in two stages: 25 stratified test episodes first, all 50 when any '
               f'of the 25 succeed or the checkpoint succeeds on training episodes. Horizon {queue.get("horizon_steps")} steps; queue paused: '
               f'{queue.get("paused")}.', '', CORRECTION, '', '**Needs a decision / blocked:**', *(asks or ['- none']), '',
               '| Stage | Run | Step | Episodes | Success | Within tol. | Geodesic m | Position m | Orientation deg | Errors |',
               '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in sorted(rows, key=lambda r: queue_rank(r['entry'])):
            entry = row['entry']
            stage = 'replay ' + entry.get('replay_split', '') if entry['split'] == 'replay' else entry['split']
            if entry.get('tag'):
                stage += f' [{entry["tag"]}: {" ".join(entry.get("server_args") or []) or "defaults"}]'
            out.append(f'| {stage} | `{entry["run"]}` | {entry.get("step") or "—"} | {row["done"]}/{row["expected"]}'
                       f'{"" if row["complete"] else " (running)"} | {fmt(row["success"], 2)} | {fmt(row["within_tolerance"], 2)} | '
                       f'{fmt(row["geodesic_distance_m"], 2)} | {fmt(row["position_error_m"], 2)} | '
                       f'{fmt(row["orientation_error_deg"], 1)} | {row["errors"]} |')
        table, waiting, chosen = execution
        out += ['', '**Execution-setting test** (training episodes; each setting against the default chunked 16 actions per '
                'query on the same checkpoint and episodes; final error = max(position / 0.05 m, orientation / 0.1 rad), '
                'geometric mean ratio to the default, below 1 is better):', '',
                '| Setting | Paired episodes | Successes (setting / default) | Final error vs default | 95 % CI | Checkpoints |',
                '| --- | ---: | ---: | ---: | --- | --- |']
        for setting, t in sorted(table.items()):
            ratio = '—' if t['mean'] is None else f'{math.exp(t["mean"]):.2f}x'
            ci = '—' if not t['ci'] else f'[{math.exp(t["ci"][0]):.2f}, {math.exp(t["ci"][1]):.2f}]'
            out.append(f'| {setting} | {t["n"]} | {t["success"]} / {t["base_success"]} | {ratio} | {ci} | {", ".join(t["checkpoints"])} |')
        if not table:
            out.append('| — | 0 | — | — | — | no results yet |')
        out += ['', (f'eval50 execution setting: **{chosen["setting"]}** (decided {chosen["decided_at"]}): {chosen["reason"]}.'
                     if chosen else 'eval50 execution setting: not decided; waiting for ' + ('; '.join(waiting) if waiting else
                                                                                           'enough paired episodes') + '.')]
        total = load['hours_queued'] + load['hours_projected']
        out += ['', f'**Lab load:** {load["queued"]} episodes queued (≈ {load["hours_queued"]:.1f} h) plus {load["projected"]} '
                f'planned for checkpoints not on the Hub yet (≈ {load["hours_projected"]:.1f} h): ≈ {total:.1f} h at '
                f'{load["minutes"]:.1f} min per episode ({load["source"]}); training episodes count only when mapped '
                f'({load["mapped"]} mapped). The lab\'s own tests outside the queue are not included.'
                + (f' **Over {LAB_HOURS_WARN:.0f} h.**' if total > LAB_HOURS_WARN else '')]
        out += ['', '**Paired comparisons:**', *(comparisons or ['- none yet (needs two complete entries on the same episodes)'])]
        out += ['', push_line]
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
        from isg_wristshuf import render as wrist_control
        text = REPORT.read_text()
        sections = [(coord_text, '<!-- coord:begin -->', '<!-- coord:end -->', 'top')]
        try:
            body, self.wrist_control_final = wrist_control(manifests[0]['runs_root'])
        except Exception as exc:  # keep the cycle alive; show the failure in the report
            body, self.wrist_control_final = f'_Wrist-goal control section failed: {type(exc).__name__}: {exc}_', None
        sections.append((f'{WRIST_BEGIN}\n{body}\n{WRIST_END}', WRIST_BEGIN, WRIST_END, 'after-coord'))
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
            elif place == 'after-coord' and '<!-- coord:end -->' in text:
                at = text.index('<!-- coord:end -->') + len('<!-- coord:end -->')
                text = text[:at] + '\n\n' + body + text[at:]
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
        files = {'docs/plan.md': PLAN, 'docs/report.md': REPORT, 'm1/decisions.md': self.m1 / 'decisions.md',
                 'm1/tierB-queue.json': self.m1 / 'tierB-queue.json'}
        if SPEC.exists():
            files['docs/correction-data-spec.md'] = SPEC
        changed = {remote: local for remote, local in files.items() if digest(local) != self.state['uploaded'].get(remote)}
        if not changed:
            return []
        return self.upload_files(changed)

    def upload_files(self, files):
        """Upload {remote path: local file} in one commit of the coordination repo (never a token-shaped string)."""
        from huggingface_hub import CommitOperationAdd
        for remote, local in files.items():
            if re.search(r'\bhf_[A-Za-z0-9]{30,}\b|\bghp_[A-Za-z0-9]{30,}\b', Path(local).read_text()):
                raise RuntimeError(f'Refusing to upload {remote}: it contains a token-shaped string')
        self.api.create_commit(REPO, repo_type='dataset', operations=[CommitOperationAdd(r, str(l)) for r, l in files.items()],
                               commit_message=f'm1 {iso(utc_now())}: ' + ', '.join(sorted(files)))
        for remote, local in files.items():
            self.state['uploaded'][remote] = digest(local)
        return sorted(files)

    def cycle(self):
        now = utc_now()
        self.decisions, self.escalations, self.first_frame_diffs = [], [], []
        download(self.api, self.inbox)
        manifests = [load(p) for p in self.manifest_paths]
        self.github = github_state(self.manifest_paths)
        self.local_runs = {name for manifest in manifests for name in manifest['_runs']}
        push_line, push_problem = push_status(self.github)
        if push_problem:
            self.escalations.append(push_problem)
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
        execution = self.exec_decision(queue, rows, manifests)
        queue['entries'].sort(key=queue_rank)
        if json.dumps(queue, sort_keys=True) != before:
            queue['updated_at'] = iso(now)
            queue_path.write_text(json.dumps(queue, indent=2) + '\n')
        lab_hours = self.lab_load(queue, rows, manifests, read_json(self.inbox / 'lab/env.json'))
        self.render_report(self.coord_section(now, m2, lab, health, calibration_text, kill_rows, trust, queue, rows,
                                              self.comparisons(rows, valid), copied, execution, lab_hours, push_line), manifests)
        if self.wrist_control_final:
            self.log_plan('wrist-control:final', 'wrist-goal control', self.wrist_control_final,
                          'held-out Tier A at the final steps; docs/report.md "Wrist-goal control"', 'relocalization L1_own δ = 0.0066')
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
    parser.add_argument('--decide', metavar='AUDIENCE', choices=['M2', 'lab', 'all'],
                        help='Post one decision through the GitHub push check, upload m1/decisions.md and the plan, exit')
    parser.add_argument('--line', action='append', default=[], help='A bullet of the --decide decision (repeat)')
    parser.add_argument('--log', help='Plan decision-log text of the --decide decision (default: its first bullet)')
    args = parser.parse_args()
    from huggingface_hub import HfApi
    manifests = args.manifest or [CHECKOUT / f'waves/{n}.json' for n in ('wave1', 'wave2', 'reloc1')]
    collector = Collector(HfApi(token=os.environ.get('HF_TOKEN')), manifests)
    if args.decide:
        if not args.line:
            parser.error('--decide needs at least one --line')
        collector.github = github_state(collector.manifest_paths)
        collector.local_runs = {name for path in collector.manifest_paths for name in load(path)['_runs']}
        key = 'manual:' + hashlib.sha256('\n'.join([args.decide, *args.line]).encode()).hexdigest()[:16]
        if not collector.decide(key, args.decide, args.line, args.log or args.line[0][:150]):
            print('Not posted: ' + ('; '.join(collector.escalations) or 'this decision is already recorded'), file=sys.stderr)
            return 1
        collector.upload_files({'m1/decisions.md': collector.m1 / 'decisions.md', 'docs/plan.md': PLAN})
        print('posted and uploaded m1/decisions.md and docs/plan.md')
        return 0
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
