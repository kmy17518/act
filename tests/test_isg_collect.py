"""Machine 1's collector on synthetic helper data: copying, rules, decisions and report sections (no Hub access)."""

from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/b1k'))
import isg_collect  # noqa: E402

TASKS = list(isg_collect.GOAL_TASKS) + [isg_collect.CONTROL_TASK]


def tier_a_result(step, l1, gap):
    tasks = {t: {'L1_own': l1, 'gap': gap if t != isg_collect.CONTROL_TASK else 0.01} for t in TASKS}
    return {'step': step, 'heldout': {'mean': {'L1_own': l1, 'gap': gap}, 'tasks': tasks}, 'train': {'mean': {}}}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'
    common = {'format': 'isg-wave/v1', 'runs_root': str(runs), 'logs_root': str(tmp_path / 'logs'),
              'staging_root': str(tmp_path / 'staging'), 'tmux_server': 'test', 'hf_owner': 'x', 'hf_repo_prefix': 'y-',
              'tier_a': {'dir': 'tierA-eval50'}, 'common': {'max_steps': 50000, 'eval_every': 10000, 'flags': []}}
    wave1 = dict(common, wave=1, runs=[{'name': n, 'seed': 0, 'flags': []} for n in
                                       ('w1-early-copy05-lrg1e-3-s0', 'w1-late-tagzero-lrg1e-4-s0', 'w1-late-tagzero-lrg1e-4-s1')])
    wave2 = dict(common, wave=2, base={'runs': ['w1-early-copy05-lrg1e-3-s0']},
                 runs=[{'name': 'w2-m2-good', 'machine': 'M2', 'seed': 0, 'flags': []},
                       {'name': 'w2-m2-bad', 'machine': 'M2', 'seed': 0, 'flags': []},
                       {'name': 'w2-m1-local', 'machine': 'M1', 'seed': 0, 'flags': []}])
    write(tmp_path / 'wave1.json', wave1)
    write(tmp_path / 'wave2.json', wave2)
    for step, (base, s0, s1) in {20000: (0.060, 0.07, 0.08), 40000: (0.055, 0.063, 0.072), 50000: (0.054, 0.062, 0.071)}.items():
        write(runs / 'w1-early-copy05-lrg1e-3-s0/tierA-eval50' / f'step_{step:08d}.json', tier_a_result(step, base, 0.05))
        write(runs / 'w1-late-tagzero-lrg1e-4-s0/tierA-eval50' / f'step_{step:08d}.json', tier_a_result(step, s0, 0.06))
        write(runs / 'w1-late-tagzero-lrg1e-4-s1/tierA-eval50' / f'step_{step:08d}.json', tier_a_result(step, s1, 0.07))
    write(runs / 'w1-early-copy05-lrg1e-3-s0/metrics.jsonl',
          ''.join(json.dumps({'step': s, 'l1': 0.02, 'timing/step_s': 0.337}) + '\n' for s in range(19501, 20001)))
    write(runs / 'w2-m1-local/metrics.jsonl', '{"step": 1, "l1": 1.0, "timing/step_s": 0.3}\n')
    coord = tmp_path / 'coord'
    plan, report = tmp_path / 'plan.md', tmp_path / 'report.md'
    write(plan, '# plan\n\n| Date | Wave | Decision | Evidence | Noise |\n| --- | --- | --- | --- | --- |\n')
    write(report, '# Report\n\nbody\n')
    write(coord / 'm1/decisions.md', '# Decisions\n')
    queue = {'entries': [{'run': r, 'step': 50000} for r in ('w1-early-copy05-lrg1e-3-s0', 'w1-late-tagzero-lrg1e-4-s0')]}
    write(coord / 'm1/tierB-queue.json', queue)
    monkeypatch.setattr(isg_collect, 'COORD', coord)
    monkeypatch.setattr(isg_collect, 'PLAN', plan)
    monkeypatch.setattr(isg_collect, 'REPORT', report)
    monkeypatch.setattr(isg_collect, 'download', lambda api, inbox: inbox.mkdir(parents=True, exist_ok=True))
    monkeypatch.setattr(isg_collect.Collector, 'upload', lambda self: [])
    collector = isg_collect.Collector(None, tmp_path / 'wave1.json', tmp_path / 'wave2.json')
    return collector, coord / 'in', runs, plan, report, coord


def test_collector_applies_the_rules_once_and_never_touches_local_runs(setup):
    collector, inbox, runs, plan, report, coord = setup
    now = isg_collect.iso(isg_collect.utc_now())
    write(inbox / 'm2/status.json', {'updated_at': now, 'needs_decision': ['which seed next?'],
                                     'runs': {'w2-m2-good': {'state': 'running', 'step': 20500, 'train_l1_last500': 0.02,
                                                             's_per_step': 0.34},
                                              'w2-m2-bad': {'state': 'running', 'step': 20500, 'train_l1_last500': 0.05,
                                                            's_per_step': 0.34}}})
    write(inbox / 'm2/calibration.json', {'machine1_reference': {'mean_train_l1_501_1000': 0.12560},
                                          'gpus': ['B200'] * 4, 'measured': {'mean_train_l1_501_1000': 0.12571},
                                          'result': 'PASS'})
    write(inbox / 'm2/tierA-eval50/w2-m2-good/step_00020000.json', tier_a_result(20000, 0.061, 0.05))
    write(inbox / 'm2/tierA-eval50/w2-m2-bad/step_00020000.json', tier_a_result(20000, 0.070, 0.001))
    write(inbox / 'm2/tierA-eval50/w2-m1-local/step_00020000.json', tier_a_result(20000, 0.5, 0.5))
    old = isg_collect.iso(isg_collect.utc_now() - timedelta(hours=2))
    write(inbox / 'lab/status.json', {'updated_at': old, 'blocked': 'simulator crashed'})
    write(inbox / 'lab/instances.json', {str(e): {'task': TASKS[e // 50], 'instance': f'i{e}', 'method': 'records',
                                                  'first_frame_mean_abs_diff': 2.5 + e % 7} for e in range(200)})
    write(inbox / 'lab/env.json', {'server': {'action_horizon': 16}, 'horizons': {'reloc': 900}})
    for run, success in (('w1-early-copy05-lrg1e-3-s0', 1), ('w1-late-tagzero-lrg1e-4-s0', 0)):
        base = inbox / 'lab/tierB' / run / 'step_00050000'
        for e in range(200):
            write(base / TASKS[e // 50] / f'episode_{e:06d}.json',
                  {'success': success if e % 10 else 1 - success, 'error': None,
                   'score': {'geodesic_m': 1.0 - success, 'angle_error_deg': 5.0 * (1 - success), 'bucket': 3}})
        write(base / 'summary.json', {'success_rate': 0.9 if success else 0.1})
    collector.cycle()
    log = (coord / 'm1/decisions.md').read_text()
    assert 'Calibration confirmed' in log and '+0.09 %' in log and '0.12571' in log
    assert '`w2-m2-good` at 20k: keep' in log and '`w2-m2-bad` at 20k: **kill**' in log
    assert 'lab is flagged as down' in log and 'Lab results are trusted from now on' in log
    assert '`w1-early-copy05-lrg1e-3-s0` beats every other finalist' in log
    assert log.count('\n## ') == 7 and all(' — to ' in line for line in log.splitlines() if line.startswith('## '))
    assert (runs / 'w2-m2-good/tierA-eval50/step_00020000.json').exists()
    assert not (runs / 'w2-m1-local/tierA-eval50').exists()  # a locally trained run is never touched
    text = report.read_text()
    assert text.index('<!-- coord:begin -->') < text.index('body') and 'which seed next?' in text
    assert 'simulator crashed' in text and '**DOWN**' in text
    assert plan.read_text().count('| coordination |') == 7  # the six decisions and the first-frame rule
    assert 'first-frame check' in text.lower() and 'score' in text
    collector.cycle()  # a second cycle on the same inputs records nothing new
    assert (coord / 'm1/decisions.md').read_text() == log


def test_failed_calibration_and_untrusted_lab(setup):
    collector, inbox, runs, plan, report, coord = setup
    write(inbox / 'm2/calibration.json', {'mean_train_l1_501_1000': 0.1270})
    write(inbox / 'lab/instances.json', {str(e): {'task': TASKS[e // 50], 'instance': f'i{e}', 'method': 'records',
                                                  'first_frame_mean_abs_diff': 3.0 if e < 150 else 41.0} for e in range(200)})
    write(inbox / 'lab/env.json', {'action_horizon': 16, 'smoke': {'action_horizon': 8}})
    collector.cycle()
    log = (coord / 'm1/decisions.md').read_text()
    assert 'Calibration failed' in log and 'must not start' in log
    assert 'trusted from now on' not in log  # never trusted, so no trust change is recorded
    write(inbox / 'm2/calibration.json', {'mean_train_l1_501_1000': 0.1259, 'pass': True})  # +0.24 %: agrees
    write(inbox / 'm2/calibration.json', {'mean_train_l1_501_1000': 0.1300, 'pass': True})  # disagrees: escalated
    collector.cycle()
    assert 'machine 1 review' in report.read_text() and (coord / 'm1/decisions.md').read_text().count('Calibration') == 1
    assert 'verifies 150/200' in report.read_text() and 'no single fixed action horizon' in report.read_text()
