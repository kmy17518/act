#!/usr/bin/env python3
"""Launch and watch the runs of an ISG goal-conditioning wave manifest (waves/wave<N>.json).

    isg_wave.py train   MANIFEST RUN [RUN ...]   one tmux session per run (trainer; resumes from latest.pt)
    isg_wave.py monitor MANIFEST [RUN ...]       session <server>:isg-monitor, one uploader window per run
                                                 (dedicated HF repo <owner>/<prefix><run>: eval/step-*.pt every
                                                 eval_every steps and exactly one resume/step-*.pt) and the
                                                 Tier A watcher window
    isg_wave.py status  MANIFEST                 one line per run: step, s/step, L1, upload, Tier A
    isg_wave.py command MANIFEST RUN             print the trainer command

Every process gets `source /tmp/dev/env.sh`; logs and exit codes go to <logs_root>/<run>.{log,exit} and
<logs_root>/<run>.upload.{log,exit}. A run with `after` waits for that run's exit file before it starts.
"""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

CHECKOUT = Path(__file__).resolve().parents[2]
PYTHON = CHECKOUT / '.venv/bin/python'
CPATH = '/tmp/dev/sysroots/libpython3.10-dev/usr/include/python3.10:/tmp/dev/sysroots/libpython3.10-dev/usr/include'


def load(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('format') != 'isg-wave/v1':
        raise SystemExit(f'{path}: not an isg-wave/v1 manifest')
    manifest['_path'] = str(Path(path).resolve())
    manifest['_runs'] = {run['name']: run for run in manifest['runs'] + manifest.get('deferred', [])}
    return manifest


def run_of(manifest, name):
    if name not in manifest['_runs']:
        raise SystemExit(f'Unknown run {name}; the manifest lists {sorted(manifest["_runs"])}')
    return manifest['_runs'][name]


def paths(manifest, name):
    return {'run': Path(manifest['runs_root']) / name, 'staging': Path(manifest['staging_root']) / name,
            'log': Path(manifest['logs_root']) / f'{name}.log', 'exit': Path(manifest['logs_root']) / f'{name}.exit',
            'upload_log': Path(manifest['logs_root']) / f'{name}.upload.log',
            'upload_exit': Path(manifest['logs_root']) / f'{name}.upload.exit'}


def max_steps(manifest, run):
    return int(run.get('max_steps', manifest['common']['max_steps']))


def repo_id(manifest, name):
    return f'{manifest["hf_owner"]}/{manifest["hf_repo_prefix"]}{name}'


def train_command(manifest, name):
    run, where = run_of(manifest, name), paths(manifest, name)
    command = [str(PYTHON), '-u', 'scripts/b1k/train_b1k.py', '--dataset-path', manifest['dataset'],
               '--task-names', *manifest['tasks'], '--episode-split', manifest['train_split'],
               '--frame-cache', manifest['frame_cache'], *manifest['common']['flags'],
               '--max-steps', str(max_steps(manifest, run)), '--seed', str(run['seed']), *run['flags'],
               '--output-dir', str(where['run']), '--wandb-entity', manifest['wandb']['entity'],
               '--wandb-project', manifest['wandb']['project'], '--wandb-id', name, '--wandb-name', name]
    return command


def bash_header():
    lines = ['set -uo pipefail', 'source /tmp/dev/env.sh', f'cd {shlex.quote(str(CHECKOUT))}',
             'export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ARROW_NUM_THREADS=1',
             'export PYTORCH_ALLOC_CONF=expandable_segments:True', f'export CPATH={CPATH}',
             'export WANDB_BASE_URL=https://api.wandb.ai HF_HUB_DISABLE_PROGRESS_BARS=1']
    return lines


def train_script(manifest, name):
    run, where = run_of(manifest, name), paths(manifest, name)
    command = ' '.join(shlex.quote(part) for part in train_command(manifest, name))
    lines = bash_header()
    if run.get('after'):
        previous = paths(manifest, run['after'])['exit']
        lines += [f'echo "waiting for {run["after"]} ({previous})"',
                  f'while [ ! -f {shlex.quote(str(previous))} ]; do sleep 20; done']
    lines += [f'export CUDA_VISIBLE_DEVICES={int(run["gpu"])}',
              f'resume=(); [ -e {shlex.quote(str(where["run"] / "latest.pt"))} ] && '
              f'resume=(--resume {shlex.quote(str(where["run"] / "latest.pt"))})',
              f'echo "$(date -u +%FT%TZ) start {name} commit $(git rev-parse HEAD) ${{resume[*]}}" >> {shlex.quote(str(where["log"]))}',
              f'taskset -c {run["cores"]} {command} "${{resume[@]}}" >> {shlex.quote(str(where["log"]))} 2>&1',
              'rc=$?', f'echo "$rc" > {shlex.quote(str(where["exit"]))}.tmp && mv {shlex.quote(str(where["exit"]))}.tmp '
              f'{shlex.quote(str(where["exit"]))}', 'echo "trainer exited with $rc"', 'exit $rc']
    return '\n'.join(['#!/usr/bin/env bash', *lines, ''])


def upload_script(manifest, name):
    run, where = run_of(manifest, name), paths(manifest, name)
    wandb = f'https://wandb.ai/{manifest["wandb"]["entity"]}/{manifest["wandb"]["project"]}/runs/{name}'
    command = [str(PYTHON), '-u', 'scripts/b1k/upload_checkpoints.py', '--run-dir', str(where['run']),
               '--staging-dir', str(where['staging']), '--repo-id', repo_id(manifest, name), '--run-id', name,
               '--max-steps', str(max_steps(manifest, run)), '--eval-every', str(manifest['common']['eval_every']),
               '--poll-seconds', '60', '--metadata', 'policy=ACT', '--metadata', f'run={name}',
               '--metadata', f'wave={manifest["wave"]}', '--metadata', 'study=isg-goal-conditioning',
               '--metadata', f'wandb_url={wandb}']
    lines = bash_header() + [
        "export CUDA_VISIBLE_DEVICES=''",
        # exit 1 is a retryable failure the uploader already journaled; 2 is a safety stop that needs a human
        'while true; do',
        f'  taskset -c 120-143 {" ".join(shlex.quote(part) for part in command)} >> {shlex.quote(str(where["upload_log"]))} 2>&1',
        '  rc=$?; [ "$rc" -eq 1 ] || break; sleep 120', 'done',
        f'echo "$rc" > {shlex.quote(str(where["upload_exit"]))}', 'echo "uploader exited with $rc"', 'exit $rc']
    return '\n'.join(['#!/usr/bin/env bash', *lines, ''])


def tmux(manifest, *args, check=True):
    return subprocess.run(['tmux', '-L', manifest['tmux_server'], *args], check=check, capture_output=True, text=True)


def sessions(manifest):
    result = tmux(manifest, 'list-sessions', '-F', '#{session_name}', check=False)
    return set(result.stdout.split()) if result.returncode == 0 else set()


def windows(manifest, session):
    result = tmux(manifest, 'list-windows', '-t', session, '-F', '#{window_name}', check=False)
    return set(result.stdout.split()) if result.returncode == 0 else set()


def write_script(manifest, name, kind, text):
    path = Path(manifest['logs_root']) / f'{name}.{kind}.sh'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def train(manifest, names):
    running = sessions(manifest)
    for name in names:
        run_of(manifest, name)
        if name in running:
            print(f'{name}: tmux session already exists; not starting another trainer')
            continue
        exit_file = paths(manifest, name)['exit']
        exit_file.unlink(missing_ok=True)
        script = write_script(manifest, name, 'train', train_script(manifest, name))
        tmux(manifest, 'new-session', '-d', '-s', name, f'bash {shlex.quote(str(script))}; exec bash')
        print(f'{name}: started in tmux -L {manifest["tmux_server"]} session {name} ({script})')


def monitor(manifest, names):
    session = 'isg-monitor'
    if session not in sessions(manifest):
        tmux(manifest, 'new-session', '-d', '-s', session, '-n', 'shell')
    present = windows(manifest, session)
    for name in names:
        if not run_of(manifest, name).get('upload', True):
            continue
        if name in present:
            print(f'{name}: uploader window already exists')
            continue
        script = write_script(manifest, name, 'upload', upload_script(manifest, name))
        tmux(manifest, 'new-window', '-d', '-t', session, '-n', name, f'bash {shlex.quote(str(script))}; exec bash')
        print(f'{name}: uploader -> {repo_id(manifest, name)} in {session}:{name}')
    if 'tier-a' not in present:
        command = (f'source /tmp/dev/env.sh; cd {shlex.quote(str(CHECKOUT))}; export CUDA_VISIBLE_DEVICES= ; '
                   f'{PYTHON} -u scripts/b1k/isg_tier_a_watch.py {shlex.quote(manifest["_path"])} '
                   f'>> {shlex.quote(str(Path(manifest["logs_root"]) / "tier-a-watch.log"))} 2>&1; exec bash')
        tmux(manifest, 'new-window', '-d', '-t', session, '-n', 'tier-a', command)
        print(f'Tier A watcher in {session}:tier-a')


def last_metrics(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 200000))
        lines = stream.read().decode(errors='replace').splitlines()
    records = []
    for line in lines[1:] if size > 200000 else lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def status(manifest):
    for name, run in manifest['_runs'].items():
        where = paths(manifest, name)
        records = last_metrics(where['run'] / 'metrics.jsonl')
        if not records:
            state = 'not started' if not where['run'].exists() else 'starting'
            print(f'{name:44s} {state}')
            continue
        last = records[-1]
        recent = records[-100:]
        step_s = sum(r['timing/step_s'] for r in recent[1:]) / max(1, len(recent) - 1)
        l1 = sum(r['l1'] for r in recent) / len(recent)
        exit_code = where['exit'].read_text().strip() if where['exit'].exists() else '-'
        upload = '-'
        status_file = where['staging'] / 'status.json'
        if status_file.exists():
            value = json.loads(status_file.read_text())
            upload = f'{value["health"]} full={value["current_step"]} eval={value["eval_steps"]}'
        tier_a = sorted(p.stem for p in (where['run'] / 'tierA').glob('step_*.json') if not p.stem.endswith('.probe')) if (where['run'] / 'tierA').exists() else []
        remaining = (max_steps(manifest, run) - last['step']) * step_s / 3600
        print(f'{name:44s} step {last["step"]:6d} {step_s:.3f} s/step L1(last100) {l1:.4f} '
              f'peak {last.get("gpu/peak_reserved_bytes", 0) / 2**30:.0f} GiB eta {remaining:.1f} h exit {exit_code} | '
              f'upload {upload} | tierA {len(tier_a)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['train', 'monitor', 'status', 'command'])
    parser.add_argument('manifest')
    parser.add_argument('runs', nargs='*')
    args = parser.parse_args()
    manifest = load(args.manifest)
    if args.action == 'train':
        train(manifest, args.runs)
    elif args.action == 'monitor':
        monitor(manifest, args.runs or [run['name'] for run in manifest['runs']])
    elif args.action == 'status':
        status(manifest)
    else:
        for name in args.runs:
            print(' '.join(shlex.quote(part) for part in train_command(manifest, name)))


if __name__ == '__main__':
    sys.exit(main())
