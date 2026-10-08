"""Machine 1's collector on synthetic helper data: queue v2 funnel, rules, decisions and report (no Hub access)."""

from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/b1k'))
import isg_collect  # noqa: E402

R = isg_collect.RELOC


def tier_a_result(step, l1, gap):
    return {'step': step, 'heldout': {'mean': {'L1_own': l1, 'gap': gap}, 'tasks': {R: {'L1_own': l1, 'gap': gap}}},
            'train': {'mean': {}}}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def episode(success, geodesic, error=None):
    return {'success': success, 'error': error, 'score': {'geodesic_distance_m': geodesic, 'position_error_m': geodesic,
                                                          'orientation_error_deg': 10 * geodesic, 'within_tolerance': success}}


def github(paths, pushed=True, without=()):
    """GitHub as the collector sees it: every local run pushed unless listed in `without`."""
    runs = sorted({n for p in paths for n in isg_collect.load(p)['_runs']} - set(without))
    return {'ok': True, 'error': None, 'remote': 'a' * 40, 'head': 'a' * 40 if pushed else 'b' * 40, 'pushed': pushed,
            'unpushed': [] if pushed else ['bbbbbbb reloc1: new runs'], 'dirty': [], 'runs': runs}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    common = {'format': 'isg-wave/v1', 'runs_root': str(runs), 'logs_root': str(tmp_path / 'logs'), 'tmux_server': 't',
              'staging_root': str(tmp_path / 's'), 'hf_owner': 'x', 'hf_repo_prefix': 'y-', 'tier_a': {'dir': 'tierA-eval50'},
              'common': {'max_steps': 100000, 'eval_every': 10000, 'flags': []}}
    wave1 = dict(common, wave=1, runs=[{'name': f'w1-late-tagzero-lrg1e-4-s{s}', 'seed': s, 'flags': []} for s in (0, 1)])
    reloc = dict(common, wave='reloc1', base={'runs': ['r1-base']},
                 runs=[{'name': 'r1-base', 'machine': 'M1', 'seed': 0, 'flags': []},
                       {'name': 'r1-m2-good', 'machine': 'M2', 'seed': 0, 'flags': []},
                       {'name': 'r1-m2-bad', 'machine': 'M2', 'seed': 0, 'flags': []}])
    write(tmp_path / 'wave1.json', wave1)
    write(tmp_path / 'reloc1.json', reloc)
    for step, (s0, s1) in {40000: (0.106, 0.113), 50000: (0.107, 0.114)}.items():
        write(runs / 'w1-late-tagzero-lrg1e-4-s0/tierA-eval50' / f'step_{step:08d}.json', tier_a_result(step, s0, 0.007))
        write(runs / 'w1-late-tagzero-lrg1e-4-s1/tierA-eval50' / f'step_{step:08d}.json', tier_a_result(step, s1, 0.002))
    write(runs / 'r1-base/tierA-eval50/step_00020000.json', tier_a_result(20000, 0.100, 0.03))
    write(runs / 'r1-base/metrics.jsonl', ''.join(json.dumps({'step': s, 'l1': 0.02, 'timing/step_s': 0.337}) + '\n'
                                                  for s in range(19501, 20001)))
    coord, plan, report = tmp_path / 'coord', tmp_path / 'plan.md', tmp_path / 'report.md'
    write(plan, '# plan\n\n| Date | Wave | Decision | Evidence | Noise |\n| --- | --- | --- | --- | --- |\n')
    write(report, '# Report\n\nbody\n')
    write(coord / 'm1/decisions.md', '# Decisions\n')
    entries = [{'run': 'replay', 'step': None, 'split': 'replay', 'replay_split': 'eval50', 'episodes': [2, 3]},
               {'run': 'replay', 'step': None, 'split': 'replay', 'replay_split': 'train', 'episodes': [39]},
               {'run': 'a', 'step': 50000, 'split': 'train', 'episodes': [8, 39, 43]},
               {'run': 'b', 'step': 50000, 'split': 'train', 'episodes': [8, 39, 43]}]
    write(coord / 'm1/tierB-queue.json', {'format': 'isg-tierB-queue/v2', 'paused': False, 'horizon_steps': 1500,
                                          'entries': entries})
    for name, value in (('COORD', coord), ('PLAN', plan), ('REPORT', report)):
        monkeypatch.setattr(isg_collect, name, value)
    monkeypatch.setattr(isg_collect, 'download', lambda api, inbox: inbox.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(isg_collect.Collector, 'upload', lambda self: [])
    monkeypatch.setattr(isg_collect.Collector, 'extend_queue', lambda self, manifests, queue: [])
    monkeypatch.setattr(isg_collect, 'github_state', lambda paths: github(paths))
    return isg_collect.Collector(None, [tmp_path / 'wave1.json', tmp_path / 'reloc1.json']), coord / 'in', runs, report, coord


def test_funnel_floor_guard_kill_rule_and_copying(setup):
    collector, inbox, runs, report, coord = setup
    now = isg_collect.iso(isg_collect.utc_now())
    write(inbox / 'm2/status.json', {'updated_at': now, 'needs_decision': ['idle: assign runs'],
                                     'runs': {'r1-m2-good': {'state': 'running', 'step': 20500, 'train_l1_last500': 0.02,
                                                             's_per_step': 0.34},
                                              'r1-m2-bad': {'state': 'running', 'step': 20500, 'train_l1_last500': 0.02,
                                                            's_per_step': 0.34}}})
    write(inbox / 'm2/tierA-eval50/r1-m2-good/step_00020000.json', tier_a_result(20000, 0.101, 0.03))
    write(inbox / 'm2/tierA-eval50/r1-m2-bad/step_00020000.json', tier_a_result(20000, 0.120, 0.001))
    write(inbox / 'm2/tierA-eval50/r1-base/step_00020000.json', tier_a_result(20000, 9.9, 9.9))
    old = isg_collect.iso(isg_collect.utc_now() - timedelta(hours=2))
    write(inbox / 'lab/status.json', {'updated_at': old, 'blocked': 'simulator crashed'})
    for split, eps in (('eval50', (2, 3)), ('train', (39,))):
        for e in eps:
            write(inbox / f'lab/diagnostics/replay/{split}/episode_{e:06d}.json', episode(True, 0.05))
        write(inbox / f'lab/diagnostics/replay/{split}/summary.json', {'n': len(eps)})
    for run, successes in (('a', (1, 0, 0)), ('b', (0, 0, 0))):
        base = inbox / 'lab/tierB-train' / run / 'step_00050000'
        for e, success in zip((8, 39, 43), successes):
            write(base / R / f'episode_{e:06d}.json', episode(bool(success), 1.0 if run == 'a' else 3.0))
        write(base / 'summary.json', {'n': 3})
    collector.cycle()
    log = (coord / 'm1/decisions.md').read_text()
    queue = json.loads((coord / 'm1/tierB-queue.json').read_text())
    assert 'Positive control' in log and '3/3 episodes reach tolerance at some step' in log and queue['paused'] is False
    assert '`a` at step 50000 succeeds on training instances' in log
    assert any(e['run'] == 'a' and e['split'] == 'eval50' and e['episodes'] is None for e in queue['entries'])
    assert [e['split'] for e in queue['entries']] == sorted((e['split'] for e in queue['entries']),
                                                            key=isg_collect.SPLIT_ORDER.get)
    assert '`r1-m2-good` at 20k: keep' in log and '`r1-m2-bad` at 20k: **kill**' in log and 'lab is flagged as down' in log
    assert (runs / 'r1-m2-good/tierA-eval50/step_00020000.json').exists()
    assert json.loads((runs / 'r1-base/tierA-eval50/step_00020000.json').read_text())['heldout']['mean']['L1_own'] == 0.100
    text = report.read_text()
    assert text.index('Relocalization phase') < text.index('body') and 'idle: assign runs' in text
    assert 'success-based' in text and 'success: +0.333' in text and 'simulator crashed' in text
    collector.cycle()  # same inputs: nothing new
    assert (coord / 'm1/decisions.md').read_text() == log


def run_episode(success, position, setting='ah16'):
    meta = ({'execution_mode': 'temporal_aggregation', 'action_horizon': 1} if setting == 'ta'
            else {'execution_mode': 'chunked', 'action_horizon': int(setting[2:])})
    return {'success': success, 'error': None, 'split': 'train', 'wall_s': 300.0, 'server_metadata': meta,
            'score': {'geodesic_distance_m': position, 'position_error_m': position, 'orientation_error_deg': 2.0,
                      'within_tolerance': success}}


def exec_setup(setup, monkeypatch, ah4_setting='ah4'):
    collector, inbox, runs, report, coord = setup
    monkeypatch.setattr(isg_collect, 'MIN_PAIRS', 3)
    manifest = json.loads(Path(collector.manifest_paths[1]).read_text())
    manifest['exec_test'] = {'lab_runs': [{'run': 'b', 'step': 50000, 'settings': ['ta']}]}
    Path(collector.manifest_paths[1]).write_text(json.dumps(manifest))
    queue = json.loads((coord / 'm1/tierB-queue.json').read_text())
    queue['entries'] += [{'run': 'a', 'step': 50000, 'split': 'eval50', 'episodes': None},
                         {'run': 'b', 'step': 50000, 'split': 'eval50', 'episodes': None},
                         {'run': 'a', 'step': 50000, 'split': 'train', 'episodes': [8, 39, 43], 'tag': 'ta',
                          'server_args': ['--temporal-agg']},
                         {'run': 'a', 'step': 50000, 'split': 'train', 'episodes': [8, 39, 43], 'tag': 'ah4',
                          'server_args': ['--action-horizon', '4']}]
    write(coord / 'm1/tierB-queue.json', queue)
    for split, eps in (('eval50', (2, 3)), ('train', (39,))):
        for e in eps:
            write(inbox / f'lab/diagnostics/replay/{split}/episode_{e:06d}.json', episode(True, 0.04))
        write(inbox / f'lab/diagnostics/replay/{split}/summary.json', {'n': len(eps)})
    for run in ('a', 'b'):  # default results, with summary.json in the task directory as the lab writes it
        base = inbox / 'lab/tierB-train' / run / 'step_00050000' / R
        for e in (8, 39, 43):
            write(base / f'episode_{e:06d}.json', run_episode(False, 0.5))
        write(base / 'summary.json', {'n': 3, 'complete': True})
    for tag, run, setting, successes, position in (('ta', 'a', 'ta', (1, 1, 0), 0.06), ('ah4', 'a', ah4_setting, (0, 0, 0), 0.5),
                                                   ('ta', 'b', 'ta', (1, 0, 0), 0.07)):
        base = inbox / 'lab/diagnostics/exec' / tag / run / 'step_00050000' / R
        for e, success in zip((8, 39, 43), successes):
            write(base / f'episode_{e:06d}.json', run_episode(bool(success), 0.04 if success else position, setting))
        write(base / 'summary.json', {'n': 3, 'complete': True})
    write(inbox / 'lab/tierB/b/step_00050000' / R / 'episode_000000.json', run_episode(False, 3.0))
    write(inbox / 'lab/env.json', {'action_horizon': 16, 'parallel_evaluators': 2,
                                   'exec_tests': {'ah4': {'action_horizon': 4}, 'ta': {'action_horizon': 1}}})
    return collector, inbox, report, coord


def test_execution_settings_choose_the_eval50_setting(setup, monkeypatch):
    collector, inbox, report, coord = exec_setup(setup, monkeypatch)
    collector.cycle()
    queue = json.loads((coord / 'm1/tierB-queue.json').read_text())
    order = [(e['split'], e.get('tag')) for e in queue['entries']]
    assert order[:4] == [('replay', None), ('replay', None), ('train', 'ta'), ('train', 'ah4')]
    assert [s for s, _ in order[4:]] == ['train', 'train', 'eval50', 'eval50']
    log = (coord / 'm1/decisions.md').read_text()
    assert 'eval50 runs with execution setting **ta**' in log and 'Entries already started keep the default: `b`@50k' in log
    evals = {e['run']: e for e in queue['entries'] if e['split'] == 'eval50'}
    assert evals['a']['tag'] == 'ta-e50' and evals['a']['server_args'] == ['--temporal-agg'] and 'tag' not in evals['b']
    assert '`a` at step 50000 succeeds on training instances (2 of 3 episodes so far, execution setting ta)' in log
    text = report.read_text()
    assert 'Execution-setting test' in text and '| ta | 6 | 3 / 0 |' in text and 'eval50 execution setting: **ta**' in text
    assert 'Lab load:' in text and 'train [ta: --temporal-agg]' in text
    assert collector.lab_trust(None, json.loads((inbox / 'lab/env.json').read_text()))[1] == 16
    collector.cycle()  # decided once; no duplicate eval50 entries under the new tag
    again = json.loads((coord / 'm1/tierB-queue.json').read_text())
    assert sum(e['split'] == 'eval50' for e in again['entries']) == 2 and (coord / 'm1/decisions.md').read_text() == log


def test_wrong_server_setting_is_flagged_and_left_out(setup, monkeypatch):
    collector, inbox, report, coord = exec_setup(setup, monkeypatch, ah4_setting='ah16')
    collector.cycle()
    text = report.read_text()
    assert "ran with server setting ['ah16'], not ah4 as queued" in text
    assert 'eval50 execution setting: not decided; waiting for `a`@50k ah4 (0/3)' in text
    assert 'eval50 runs with execution setting' not in (coord / 'm1/decisions.md').read_text()


def test_training_success_with_a_complete_eval50_result_is_not_requeued(setup):
    collector, inbox, runs, report, coord = setup
    base = inbox / 'lab/tierB-train/a/step_00050000' / R
    for e, success in zip((8, 39, 43), (1, 0, 0)):
        write(base / f'episode_{e:06d}.json', run_episode(bool(success), 0.04 if success else 0.5))
    for e in range(50):
        write(inbox / 'lab/tierB/a/step_00050000' / R / f'episode_{e:06d}.json', run_episode(False, 2.0))
    collector.cycle()
    queue = json.loads((coord / 'm1/tierB-queue.json').read_text())
    assert not any(e['split'] == 'eval50' for e in queue['entries'])
    assert 'it already has a complete eval50 result (0% success), so nothing is re-queued' in (coord / 'm1/decisions.md').read_text()


def test_two_stage_eval50(setup, monkeypatch, tmp_path):
    collector, inbox, runs, report, coord = setup
    stage1 = list(range(0, 50, 2))
    write(tmp_path / 'picks.json', {'eval25': stage1})
    real_read = isg_collect.read_json
    monkeypatch.setattr(isg_collect, 'read_json', lambda path: real_read(tmp_path / 'picks.json')
                        if str(path).endswith('splits/reloc-picks.json') else real_read(path))
    queue = json.loads((coord / 'm1/tierB-queue.json').read_text())
    queue['entries'] += [{'run': run, 'step': 50000, 'split': 'eval50', 'episodes': stage1} for run in ('a', 'b', 'c')]
    write(coord / 'm1/tierB-queue.json', queue)
    for e, success in zip((8, 39, 43), (1, 0, 0)):  # a succeeds on a training episode
        write(inbox / 'lab/tierB-train/a/step_00050000' / R / f'episode_{e:06d}.json', run_episode(bool(success), 0.04 if success else 0.5))
    for e in stage1:  # b: one of its 25 stage-1 episodes succeeds; c: none
        write(inbox / 'lab/tierB/b/step_00050000' / R / f'episode_{e:06d}.json', run_episode(e == 10, 0.04 if e == 10 else 2.0))
        write(inbox / 'lab/tierB/c/step_00050000' / R / f'episode_{e:06d}.json', run_episode(False, 2.0))
    collector.cycle()
    evals = {e['run']: e['episodes'] for e in json.loads((coord / 'm1/tierB-queue.json').read_text())['entries'] if e['split'] == 'eval50'}
    assert evals == {'a': None, 'b': None, 'c': stage1}
    log = (coord / 'm1/decisions.md').read_text()
    assert '`a`@50k eval50: its checkpoint succeeds on training episodes' in log and '`b`@50k eval50: 1 of its first 25 stage-1 episodes succeed' in log


def test_replay_that_reached_tolerance_passes(setup):
    collector, inbox, runs, report, coord = setup
    boundary = {'success': False, 'error': None, 'first_success_step': 625, 'binary_success_ever': True,
                'score': {'geodesic_distance_m': 0.05, 'position_error_m': 0.0501, 'orientation_error_deg': 2.0,
                          'within_tolerance': False}}
    for split, eps in (('eval50', (2, 3)), ('train', (39,))):
        for e in eps:
            write(inbox / f'lab/diagnostics/replay/{split}/episode_{e:06d}.json', boundary if e == 39 else episode(True, 0.04))
        write(inbox / f'lab/diagnostics/replay/{split}/summary.json', {'n': len(eps)})
    collector.cycle()
    assert json.loads((coord / 'm1/tierB-queue.json').read_text())['paused'] is False
    assert '3/3 episodes reach tolerance at some step. Passed' in (coord / 'm1/decisions.md').read_text()
    never = dict(boundary, first_success_step=-1, binary_success_ever=False)
    assert isg_collect.reached_tolerance(boundary) and not isg_collect.reached_tolerance(never)


def test_decisions_naming_unpushed_runs_are_held(setup, monkeypatch):
    collector, inbox, runs, report, coord = setup
    manifest = json.loads(Path(collector.manifest_paths[1]).read_text())
    manifest['runs'].append({'name': 'r1-new-s0', 'machine': 'M2', 'seed': 0, 'flags': []})
    Path(collector.manifest_paths[1]).write_text(json.dumps(manifest))
    queue = json.loads((coord / 'm1/tierB-queue.json').read_text())
    queue['entries'].append({'run': 'r1-new-s0', 'step': 100000, 'split': 'train', 'episodes': [8]})
    write(coord / 'm1/tierB-queue.json', queue)
    write(inbox / 'lab/tierB-train/r1-new-s0/step_00100000' / R / 'episode_000008.json', run_episode(True, 0.04))
    monkeypatch.setattr(isg_collect, 'github_state', lambda paths: github(paths, pushed=False, without=('r1-new-s0',)))
    collector.cycle()
    text = report.read_text()
    assert 'r1-new-s0' not in (coord / 'm1/decisions.md').read_text()
    assert 'held until the push lands' in text and 'runs `r1-new-s0` are not in the manifests on GitHub' in text
    assert '**GitHub push check: NOT PUSHED' in text and 'bbbbbbb reloc1: new runs' in text
    monkeypatch.setattr(isg_collect, 'github_state', lambda paths: github(paths))
    collector.cycle()
    assert '`r1-new-s0` at step 100000 succeeds on training instances' in (coord / 'm1/decisions.md').read_text()
    assert 'contains machine 1\'s HEAD' in report.read_text()


def test_wrist_control_reading_and_section(setup):
    import isg_wristshuf as w
    d = w.DELTA
    assert w.reading({0: (0.085, 0.066, 0.068), 1: (0.088, 0.076, 0.079)})[0] == 'not wrist content'
    assert w.reading({0: (0.085, 0.066, 0.083), 1: (0.088, 0.076, 0.090)})[0] == 'wrist content'
    assert w.reading({0: (0.085, 0.066, 0.085 + d + 0.001), 1: (0.088, 0.076, 0.079)})[0] == 'inconclusive'
    assert w.reading({0: (0.085, 0.066, 0.068), 1: (0.088, 0.076, 0.086)})[0] == 'mixed'
    collector, inbox, runs, report, coord = setup
    collector.cycle()
    text = report.read_text()
    assert '<!-- wrist-control:begin -->' in text and 'Reading: pending' in text
    assert text.index('<!-- coord:end -->') < text.index('<!-- wrist-control:begin -->') < text.index('body')
    values = {'single-task': {0: (0.085, 0.066, 0.068), 1: (0.088, 0.076, 0.078)}}
    for study, spec in w.STUDIES.items():
        for seed, roles in spec['runs'].items():
            for role, run in roles.items():
                l1 = values.get(study, {}).get(seed, (0.09, 0.08, 0.081))[w.ROLES.index(role)]
                tasks = {task: {'L1_own': l1, 'gap': 0.02 + 0.01 * w.ROLES.index(role)} for task in w.TASKS}
                write(runs / run / 'tierA-eval50' / f'step_{spec["step"]:08d}.json', {'heldout': {'tasks': tasks}})
    collector.cycle()
    text, plan = report.read_text(), isg_collect.PLAN.read_text()
    assert text.count('<!-- wrist-control:begin -->') == 1 and 'Reading (single-task, primary): not wrist content' in text
    assert 'Wrist-goal control, final reading (single-task, primary): not wrist content' in plan
    collector.cycle()
    assert isg_collect.PLAN.read_text().count('final reading') == 1
    assert 'Wrist-goal control' not in (coord / 'm1/decisions.md').read_text()


def test_wrist_detach_follow_up_reading(setup):
    import isg_wristshuf as w
    base = {'W': 0.066, 'C': 0.091, 'R': 0.088, 'H': 0.085}
    assert w.detach_reading({0: dict(base, D=0.068), 1: dict(base, D=0.069)})[0] == 'wrist content'
    assert w.detach_reading({0: dict(base, D=0.084), 1: dict(base, D=0.083)})[0] == 'filters trained on wrist content'
    assert w.detach_reading({0: dict(base, D=0.075), 1: dict(base, D=0.069)})[0] == 'mixed'
    collector, inbox, runs, report, coord = setup
    collector.cycle()
    assert 'Follow-up reading: pending' in report.read_text()
    values = {0: dict(base, D=0.068), 1: dict(base, D=0.069)}
    for seed, roles in w.DETACH_RUNS.items():
        for key, run in roles.items():
            for step in (10000, w.DETACH_STEP):
                l1 = values[seed][key] + (0.01 if step == 10000 else 0.0)
                write(runs / run / 'tierA-eval50' / f'step_{step:08d}.json', {'heldout': {'tasks': {w.RELOC: {'L1_own': l1, 'gap': 0.02}}}})
    collector.cycle()
    text, plan = report.read_text(), isg_collect.PLAN.read_text()
    assert 'Follow-up reading (primary): wrist content' in text and '| s0 | W `r1-early-wrist-st0-s0` | 0.0760 |' in text
    assert 'gain kept without filter training (H − D) / (H − W) = 89%' in text
    assert plan.count('Wrist-goal follow-up (goal gradient stopped), final reading: wrist content') == 1
    collector.cycle()
    assert isg_collect.PLAN.read_text().count('Wrist-goal follow-up') == 1


def test_decision_transfer_reading(setup):
    import isg_xfer as x
    assert x.classify({0: -0.02, 1: -0.01}, 0.0066) == 'chosen better'
    assert x.classify({0: 0.003, 1: -0.002}, 0.0066) == 'no difference'
    assert x.classify({0: -0.02, 1: 0.001}, 0.0066) == 'mixed'
    collector, inbox, runs, report, coord = setup
    collector.cycle()
    assert '<!-- xfer:begin -->' in report.read_text() and 'Decision transfer' in report.read_text()
    l1 = {'early-wrist': 0.06, 'early-copy05': 0.08, 'early-zero': 0.081, 'late-tagzero': 0.085}
    for short, (task, pattern) in x.TASKS.items():
        for seed in x.SEEDS:
            for config, value in l1.items():
                value += 0.004 * seed  # seed noise: delta 0.004 on object scaling
                if short == 'mental rotation' and config == 'late-tagzero':
                    value = 0.09 if seed == 0 else 0.074  # seed spread 0.016 sets mental rotation's delta
                write(runs / pattern.format(config=config, seed=seed) / 'tierA-eval50' / f'step_{x.STEP:08d}.json',
                      {'heldout': {'tasks': {task: {'L1_own': value, 'gap': 0.02}}}})
    collector.cycle()
    text, plan = report.read_text(), isg_collect.PLAN.read_text()
    assert ('wrist goal views (A0 vs A1): relocalization chosen better; mental rotation chosen better; object scaling chosen '
            'better (transfers to mental rotation, object scaling)') in plan
    assert 'copy:0.5 vs zero stem init (A1 vs A2): relocalization no difference; mental rotation no difference' in plan
    assert ('early vs late fusion (A1 vs A3): relocalization no difference; mental rotation no difference; object scaling '
            'chosen better (transfers to mental rotation)') in plan
    assert '| wrist goal views (A0 vs A1) |' in text and 'object scaling (delta 0.0040)' in text
    collector.cycle()
    assert isg_collect.PLAN.read_text().count('Decision transfer, final reading') == 1


def test_failed_replay_pauses_the_queue(setup):
    collector, inbox, runs, report, coord = setup
    for split, eps in (('eval50', (2, 3)), ('train', (39,))):
        for e in eps:
            write(inbox / f'lab/diagnostics/replay/{split}/episode_{e:06d}.json', episode(e != 3, 0.05 if e != 3 else 2.0))
        write(inbox / f'lab/diagnostics/replay/{split}/summary.json', {'n': len(eps)})
    for run in ('a', 'b'):
        base = inbox / 'lab/tierB-train' / run / 'step_00050000'
        for e in (8, 39, 43):
            write(base / R / f'episode_{e:06d}.json', episode(False, 1.0 if run == 'a' else 3.0))
        write(base / 'summary.json', {'n': 3})
    collector.cycle()
    assert 'floor guard' in report.read_text() and 'geodesic_distance_m: -2.000' in report.read_text()
    assert json.loads((coord / 'm1/tierB-queue.json').read_text())['paused'] is True
    assert 'Failed: the action path' in (coord / 'm1/decisions.md').read_text()
