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
    assert 'Positive control' in log and '3/3 episodes end within tolerance' in log and queue['paused'] is False
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
