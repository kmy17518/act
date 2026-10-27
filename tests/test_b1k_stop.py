"""Stop study, Method A: on-the-fly stop / progress targets, the widened ACT head and its loss, end-window
oversampling, validation metrics, the served signals with the DONE gate, and the offline evaluation script."""

import asyncio
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1k_dataset import (ACTION_DELTA, B1KDataset, StepBatchSampler, action_reference, extract_state,  # noqa: E402
                         split_stop_signals, stop_targets)
from b1k_server import B1KServer, PolicyPredictor, Session, StopGate, packb, unpackb  # noqa: E402
from b1k_training import auroc, load_checkpoint, make_policy, stop_signal_metrics  # noqa: E402
from test_b1k import (CountingPredictor, checkpoint_stub, observation, small_model_config, tiny_root,  # noqa: E402,F401
                      write_settle_windows)
from test_b1k_side import RELQ, add_episodes, tiny_common, train_fn  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def test_stop_targets_mark_frames_from_T_end_and_progress_towards_it():
    targets = stop_targets(frame=3, program_length=5, chunk_size=4)  # chunk frames 3, 4, 5, 6
    assert targets.shape == (4, 2) and targets.dtype == np.float32
    np.testing.assert_array_equal(targets[:, 0], [-1, -1, 1, 1])
    np.testing.assert_allclose(targets[:, 1], 2 * np.array([3 / 5, 4 / 5, 1, 1]) - 1, rtol=1e-6)
    motion, signals = split_stop_signals(np.concatenate([np.zeros((2, 4, 23)), targets[None].repeat(2, 0)], axis=-1))
    assert motion.shape == (2, 4, 23) and signals.shape == (2, 4, 2)
    np.testing.assert_array_equal(signals[0, :, 0], [0, 0, 1, 1])
    assert split_stop_signals(np.zeros((2, 4, 23)))[1] is None
    with pytest.raises(ValueError, match='23 or 25 output dims'):
        split_stop_signals(np.zeros((2, 4, 24)))


def test_dataset_appends_targets_on_the_fly_without_touching_motion_dims_or_fingerprint(tiny_root):
    write_settle_windows(tiny_root, {42: 3, 99: 2})
    plain = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), settle_steps='0')
    stop = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), settle_steps='0', stop_dims=True)
    assert stop.fingerprint() == plain.fingerprint()  # the same statistics files
    assert stop.program_length(42) == 3 and stop.program_length(99) == 2 and len(stop) == 5
    stats = plain.compute_normalization_stats('quantile', 'relative')
    assert stop.compute_normalization_stats('quantile', 'relative') == stats  # 23 motion dims, as before
    plain.stats = stop.stats = stats
    for episode, program in ((42, 3), (99, 2)):
        for frame in range(program):
            _, _, expected, is_pad, _, _ = plain.sample_at(episode, frame)
            _, _, actions, is_pad_stop, _, _ = stop.sample_at(episode, frame)
            assert actions.shape == (4, 25) and expected.shape == (4, 23)
            torch.testing.assert_close(actions[:, :23], expected)
            torch.testing.assert_close(is_pad_stop, is_pad)  # the motion mask is unchanged ...
            # ... and with --settle-steps 0 the stop column is exactly the padding: stop = 1 from T_end on
            torch.testing.assert_close(actions[:, 23], 2 * is_pad.float() - 1)
            torch.testing.assert_close(actions[:, 24], 2 * torch.clamp(torch.arange(frame, frame + 4) / program, max=1).float() - 1)
    # the whole recording loaded (settle_steps all): settle frames are recorded motion targets and stop positives
    whole = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), settle_steps='all', stop_dims=True)
    whole.stats = stats
    _, _, actions, is_pad, _, _ = whole.sample_at(42, 2)  # frames 2, 3, 4 recorded (program 3), 5 padded
    torch.testing.assert_close(is_pad, torch.tensor([False, False, False, True]))
    torch.testing.assert_close(actions[:, 23], torch.tensor([-1., 1., 1., 1.]))
    for dataset in (plain, stop, whole):
        dataset.close()
    with pytest.raises(FileNotFoundError, match=r'--stop-dims needs .*settle_windows\.json'):
        (tiny_root / 'isg_meta/settle_windows.json').unlink()
        B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), stop_dims=True)


def test_end_window_oversampling_keeps_the_plain_sequence_at_zero(tiny_root):
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16))
    plain = [batch for batch in StepBatchSampler(dataset, 6, 0, 3, 0)]
    assert [batch for batch in StepBatchSampler(dataset, 6, 0, 3, 0, end_window=4, end_fraction=0.0)] == plain
    everything_at_the_end = StepBatchSampler(dataset, 6, 0, 3, 0, end_window=2, end_fraction=1.0)
    for batch in everything_at_the_end:
        for index in batch:
            position = int(np.searchsorted(dataset.ends, index, side='right'))
            assert index - dataset.starts[position] >= dataset.lengths[position] - 2
    mixed = [index for batch in StepBatchSampler(dataset, 64, 0, 4, 0, end_window=2, end_fraction=0.5) for index in batch]
    tail = [index - dataset.starts[int(np.searchsorted(dataset.ends, index, side='right'))] >= 3 for index in mixed]
    assert 0.5 < np.mean(tail) < 0.9  # half forced into the last two frames, the rest uniform over five
    with pytest.raises(ValueError, match='end_fraction'):
        StepBatchSampler(dataset, 6, 0, 3, 0, end_window=0, end_fraction=0.5)
    dataset.close()


def test_auroc_matches_pairwise_comparison_with_ties():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, 60).astype(np.float64)
    labels = rng.random(60) < 0.4
    positives, negatives = scores[labels], scores[~labels]
    pairwise = np.mean([(p > n) + 0.5 * (p == n) for p in positives for n in negatives])
    assert auroc(scores, labels) == pytest.approx(pairwise)
    assert auroc([1, 2], [True, True]) != auroc([1, 2], [True, True])  # nan without both classes
    metrics = stop_signal_metrics(np.array([[[-1, -1], [1, 0]]]), np.array([[[-1, -1], [1, 1]]]), np.array([[False, True]]))
    assert metrics['val/stop_auroc'] == 1 and metrics['val/stop_acc'] == 1 and metrics['val/stop_positive_rate'] == 0.5
    assert metrics['val/progress_mae'] == pytest.approx(0.25) and np.isnan(metrics['val/stop_auroc_valid'])


def test_policy_loss_masks_motion_dims_only_and_weights_stop_positives():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    config = dict(small_model_config(), state_dim=23, l1_normalization='valid', action_dim=25, stop_dims=True,
                  stop_pos_weight=5.0, stop_loss_weight=0.5, progress_loss_weight=0.25)
    policy = make_policy(config, 'cpu')
    assert policy.model.action_head.out_features == 25 and policy.model.encoder_action_proj.in_features == 25
    qpos = torch.randn(3, 23)
    image = torch.rand(3, 3, 3, 16, 16)
    actions = torch.randn(3, 4, 25)
    actions[..., 23] = torch.tensor([[-1, -1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, 1]]).float()
    is_pad = torch.tensor([[False, False, True, True], [False] * 4, [False, True, True, True]])
    torch.manual_seed(1)
    losses = policy(qpos, image, actions, is_pad)
    torch.manual_seed(1)
    import torchvision.transforms as transforms
    normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    a_hat, _, _ = policy.model(qpos, normalized, None, actions, is_pad)
    all_l1 = (actions - a_hat).abs()
    valid = (~is_pad).unsqueeze(-1).float()
    motion = (all_l1[..., :23] * valid).sum() / (valid.sum() * 23)
    positive = (actions[..., 23] > 0).float()
    weight = 1 + 4 * positive
    stop = (all_l1[..., 23] * weight).sum() / weight.sum()
    progress = all_l1[..., 24].mean()
    torch.testing.assert_close(losses['l1'], motion)
    torch.testing.assert_close(losses['l1_stop'], stop)
    torch.testing.assert_close(losses['l1_progress'], progress)
    torch.testing.assert_close(losses['loss'], motion + losses['kl'] * config['kl_weight'] + 0.5 * stop + 0.25 * progress)
    torch.manual_seed(1)
    _, terms = policy(qpos, image, actions, is_pad, l1_terms=True)
    torch.testing.assert_close(terms['l1_count'].float(), valid.sum((1, 2)) * 23)


def stop_run(root, tmp_path, name, extra, steps=2):
    return train_fn(tiny_common(root) + RELQ + ['--settle-steps', '0', '--stop-dims', '--task-names', 'first',
                                                 '--max-steps', str(steps), '--output-dir', str(tmp_path / name)] + extra)


def test_training_records_the_widened_head_and_validation_stop_metrics_and_resumes(tiny_root, tmp_path):
    torch.set_num_threads(1)
    add_episodes(tiny_root, 3)
    write_settle_windows(tiny_root, {42: 3, 99: 2, 50: 4, 51: 3, 52: 5})
    with pytest.raises(ValueError, match='need --stop-dims'):
        train_fn(tiny_common(tiny_root) + ['--settle-steps', '0', '--task-names', 'first', '--stop-pos-weight', '5',
                                           '--max-steps', '1', '--output-dir', str(tmp_path / 'bad')])
    path = stop_run(tiny_root, tmp_path, 'a', ['--stop-pos-weight', '5', '--stop-oversample', '0.3', '--val-fraction',
                                                '0.4', '--val-every', '1', '--val-batches', '1'])
    checkpoint = load_checkpoint(path)
    config = checkpoint['model_config']
    assert config['action_dim'] == 25 and config['stop_dims'] is True and config['stop_pos_weight'] == 5.0
    assert checkpoint['adapter_config']['action_dim'] == 23 and checkpoint['adapter_config']['stop_signals'] == ['stop', 'progress']
    assert checkpoint['model']['model.action_head.weight'].shape[0] == 25
    assert len(checkpoint['normalization']['action_q01']) == 23  # the motion normalization is untouched
    assert checkpoint['train_config']['stop_oversample'] == 0.3
    rows = [json.loads(line) for line in (tmp_path / 'a/metrics.jsonl').read_text().splitlines()]
    assert {'l1', 'l1_stop', 'l1_progress', 'kl', 'loss'} <= set(rows[-1])
    assert {'val/stop_auroc', 'val/stop_acc', 'val/progress_mae', 'val/stop_positive_rate', 'val/l1_stop',
            'val/l1_action_units'} <= set(rows[-1])
    assert rows[-1]['loss'] == pytest.approx(rows[-1]['l1'] + 10 * rows[-1]['kl'] + rows[-1]['l1_stop'] + rows[-1]['l1_progress'],
                                             rel=1e-4)
    # resume keeps the saved stop configuration without the flags; a conflicting flag is refused
    resumed = train_fn(tiny_common(tiny_root) + ['--settle-steps', '0', '--task-names', 'first', '--max-steps', '3',
                                                 '--resume', str(path), '--output-dir', str(tmp_path / 'a')])
    assert load_checkpoint(resumed)['model_config']['stop_pos_weight'] == 5.0
    with pytest.raises(ValueError, match='--stop-dims differs from checkpoint'):
        train_fn(tiny_common(tiny_root) + ['--settle-steps', '0', '--task-names', 'first', '--no-stop-dims',
                                           '--max-steps', '4', '--resume', str(resumed), '--output-dir', str(tmp_path / 'a')])


class SignalPredictor:
    """Chunks of constant actions with a programmable stop / progress schedule per call."""
    def __init__(self, stop=1.0, progress=1.0, chunk=4):
        self.stop, self.progress, self.chunk = stop, progress, chunk
        self.calls = 0

    def predict(self, qpos, images, task_ids=None, goal=None, states=None):
        self.calls += 1
        batch = len(qpos)
        actions = np.full((batch, self.chunk, 23), 0.5, dtype=np.float32)
        actions[..., 0] = 0.1  # a base command, overridden by the hold pose
        signals = np.empty((batch, self.chunk, 2), dtype=np.float32)
        signals[..., 0], signals[..., 1] = self.stop, self.progress
        return actions, signals


def stop_checkpoint():
    checkpoint = checkpoint_stub()
    checkpoint['model_config'] = {'num_queries': 4, 'stop_dims': True}
    return checkpoint


def test_session_gates_done_after_hold_steps_and_holds_the_pose():
    predictor = SignalPredictor(stop=0.8, progress=0.95)
    session = Session(predictor, stop_checkpoint(), action_horizon=2, stop_gate=StopGate(0.5, 0.9, hold_steps=3),
                      gripper_snap=None)
    obs = observation(2, np.array([7, 9]), value=0.3)
    for step in range(2):
        action = session.act(obs)
        assert session.signals['done'].tolist() == [False, False] and session.signals['hold_count'].tolist() == [step + 1] * 2
        np.testing.assert_allclose(session.signals['stop'], 0.8)
        np.testing.assert_allclose(action[:, 0], 0.1)
    action = session.act(obs)  # third consecutive step: DONE, the hold pose replaces the chunk's action
    assert session.signals['done'].tolist() == [True, True]
    expected = action_reference(extract_state(obs['robot_r1::proprio'], 'sum'), ACTION_DELTA)
    expected[:, ACTION_DELTA['gripper_action_dims']] = 0.5  # the last executed gripper command
    np.testing.assert_allclose(action, expected, atol=1e-6)
    assert action[0, 0] == 0  # zero base velocity
    predictor.stop = 0.0  # DONE is latched: later low signals change nothing, the pose stays
    later = session.act(obs)  # step 4 still executes the chunk fetched at step 3 (stop 0.8 kept for the horizon)
    np.testing.assert_allclose(later, expected, atol=1e-6)
    assert session.signals['done'].tolist() == [True, True] and session.signals['hold_count'].tolist() == [4, 4]
    later = session.act(obs)  # step 5: a new chunk with stop 0 resets the count, DONE stays
    np.testing.assert_allclose(later, expected, atol=1e-6)
    assert session.signals['done'].tolist() == [True, True] and session.signals['hold_count'].tolist() == [0, 0]
    session.reset()
    assert session.act(obs)[0, 0] == pytest.approx(0.1) and session.signals['done'].tolist() == [False, False]
    # a disabled gate reports and counts but never declares DONE
    silent = Session(SignalPredictor(), stop_checkpoint(), action_horizon=1, stop_gate=StopGate(hold_steps=1, enabled=False))
    for _ in range(3):
        assert silent.act(obs)[0, 0] == pytest.approx(0.1) and silent.signals['done'].tolist() == [False, False]
    assert silent.signals['hold_count'].tolist() == [3, 3]
    # a predictor without signals (any earlier checkpoint) serves as before
    plain = Session(CountingPredictor(), checkpoint_stub(), 2)
    assert plain.act(observation(2, np.array([7, 9]))).shape == (2, 23) and plain.signals is None


def test_chunked_signals_summarize_the_first_positions_and_task_changes_reset_the_count():
    class RampPredictor(SignalPredictor):
        def predict(self, qpos, images, task_ids=None, goal=None, states=None):
            actions, signals = super().predict(qpos, images, task_ids, goal, states)
            signals[..., 0] = np.linspace(0, 1, self.chunk)  # 0, 1/3, 2/3, 1 over the chunk
            return actions, signals
    session = Session(RampPredictor(), stop_checkpoint(), action_horizon=2, stop_positions=3,
                      stop_gate=StopGate(0.3, 0.9, hold_steps=2))
    obs = observation(1, 7)
    session.act(obs)
    assert session.signals['stop'][0] == pytest.approx(1 / 3)  # mean of the first three positions, kept for the chunk
    session.act(obs)
    assert session.signals['hold_count'][0] == 2 and session.signals['done'][0]
    with pytest.raises(ValueError, match='--stop-positions'):
        Session(RampPredictor(), stop_checkpoint(), action_horizon=2, stop_positions=5)
    with pytest.raises(ValueError, match='--stop-positions'):
        Session(RampPredictor(), stop_checkpoint(), action_horizon=1, temporal_agg=True, stop_positions=2)
    # the count restarts when a slot's task changes
    session = Session(SignalPredictor(), stop_checkpoint(), action_horizon=1, stop_gate=StopGate(hold_steps=3))
    session.act(observation(1, 7))
    session.act(observation(1, 7))
    session.act(observation(1, 9))
    assert session.signals['hold_count'].tolist() == [1] and not session.signals['done'][0]


def test_temporal_aggregation_averages_the_signal_columns_like_the_actions():
    predictor = SignalPredictor(stop=1.0, progress=1.0)
    session = Session(predictor, stop_checkpoint(), action_horizon=1, temporal_agg=True, stop_gate=StopGate(hold_steps=2))
    obs = observation(1, 7)
    session.act(obs)
    np.testing.assert_allclose(session.signals['stop'], 1.0)
    predictor.stop = 0.0  # the new chunk says 0, the older overlapping chunk still says 1: upstream weights average them
    session.act(obs)
    weights = np.exp(-0.01 * np.arange(2))
    np.testing.assert_allclose(session.signals["stop"], weights[0] / weights.sum(), rtol=1e-6)  # oldest chunk first


def test_server_payload_and_metadata_carry_the_signals(tmp_path):
    server = B1KServer(stop_checkpoint(), SignalPredictor(), action_horizon=1, gripper_snap=None,
                       stop_gate=StopGate(hold_steps=2), signal_log=str(tmp_path / 'signals.jsonl'))
    assert server.metadata['stop_signals'] == {'response_keys': ['stop', 'progress', 'done'], 'stop_positions': 1,
                                               'stop_threshold': 0.5, 'progress_threshold': 0.9, 'hold_steps': 2,
                                               'enabled': True, 'done_action': 'hold_pose'}

    class FakeSocket:
        def __init__(self, messages):
            self.messages, self.sent = messages, []

        async def send(self, data):
            self.sent.append(unpackb(data))

        async def close(self, code=None, reason=None):
            pass

        def __aiter__(self):
            async def generate():
                for message in self.messages:
                    yield message
            return generate()

    socket = FakeSocket([packb(observation(1, 7)), packb(observation(1, 7)), packb({'reset': True}), packb(observation(1, 7))])
    asyncio.run(server.handler(socket))
    metadata, first, second, third = socket.sent
    assert metadata['stop_signals']['hold_steps'] == 2
    assert first['done'].tolist() == [False] and second['done'].tolist() == [True] and third['done'].tolist() == [False]
    assert first['stop'].dtype == np.float32 and first['action'].shape == (1, 23)
    log = [json.loads(line) for line in (tmp_path / 'signals.jsonl').read_text().splitlines()]
    assert [(row['session'], row['step'], row['done']) for row in log] == [(1, 1, False), (1, 2, True), (2, 1, False)]
    plain = B1KServer(checkpoint_stub(), CountingPredictor(), action_horizon=1)
    assert plain.metadata['stop_signals'] is None


def test_offline_evaluation_replays_validation_episodes_through_the_session(tiny_root, tmp_path):
    torch.set_num_threads(1)
    add_episodes(tiny_root, 3)
    write_settle_windows(tiny_root, {42: 3, 99: 2, 50: 4, 51: 3, 52: 5})
    path = stop_run(tiny_root, tmp_path, 'run', ['--val-fraction', '0.4', '--goal-fusion', 'late', '--goal-views', 'zed_link'])
    output = tmp_path / 'eval'
    python = sys.executable
    result = subprocess.run([python, str(REPO / 'scripts/b1k/eval_stop_offline.py'), '--checkpoint', str(path),
                             '--device', 'cpu', '--slots', '2', '--action-horizon', '2', '--stop-hold-steps', '1',
                             '--stop-threshold', '0.0', '--progress-threshold', '0.0', '--output', str(output)],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'RESULT:' in result.stdout
    summary = json.loads((output / 'summary.json').read_text())
    val_episodes = load_checkpoint(path)['val_split']['val_episodes']
    assert summary['episodes'] == len(val_episodes) and set(summary['methods']) == {'A', 'baseline'}
    a = summary['methods']['A']
    assert a['done_rate'] == 1.0 and a['false_stop_rate'] == 1.0  # thresholds 0: DONE at the first step of every episode
    assert 0 <= a['frame_auroc_stop'] <= 1 or np.isnan(a['frame_auroc_stop'])
    assert len(summary['baseline_grid']) == 16 and {'eps', 'steps', 'false_stop_rate'} <= set(summary['methods']['baseline'])
    rows = [json.loads(line) for line in (output / 'episodes.jsonl').read_text().splitlines()]
    assert {row['episode'] for row in rows} == set(val_episodes) and all(row['A_done_step'] == 0 for row in rows)
    traces = np.load(output / 'traces.npz')
    for row in rows:
        assert len(traces[f"stop_{row['episode']}"]) == row['length'] == 5  # every recorded frame, the settle tail included
        assert row['program_length'] < row['length']
    # the same checkpoint served with the real predictor gives finite signals in [0, 1]
    checkpoint = load_checkpoint(path)
    predictor = PolicyPredictor(checkpoint, 'cpu')
    actions, signals = predictor.predict(torch.zeros(1, checkpoint['model_config']['state_dim']), torch.rand(1, 3, 3, 16, 16),
                                         np.array([7]), torch.rand(1, 1, 3, 16, 16), np.zeros((1, 61), dtype=np.float32))
    assert actions.shape == (1, 4, 23) and signals.shape == (1, 4, 2) and signals.min() >= 0 and signals.max() <= 1

    # Method B on the same episodes: features from a tiny frame cache, train, eval, and the A + B combination
    from b1k_frame_cache import build_frame_cache
    cache = tmp_path / 'cache'
    build_frame_cache(B1KDataset(tiny_root, chunk_size=1, image_size=(16, 16), task_names=['first']), cache, log=lambda *_: None)
    script = REPO / 'scripts/b1k/stop_verifier.py'
    features, verifier, b_out, ab_out = tmp_path / 'features', tmp_path / 'verifier.pt', tmp_path / 'b', tmp_path / 'ab'
    common = ['--device', 'cpu']
    for argv in (['extract', '--dataset-path', str(tiny_root), '--frame-cache', str(cache), '--task-names', 'first',
                  '--episode-split', 'none', '--val-fraction', '0.4', '--image-size', '16', '16', '--stride', '1',
                  '--backbone', 'tiny', '--batch-size', '3', '--output', str(features)],
                 ['train', '--features', str(features), '--output', str(verifier), '--history', '2', '--gap', '1',
                  '--hidden', '16', '--epochs', '3', '--batch-size', '4'],
                 ['eval', '--features', str(features), '--verifier', str(verifier), '--query-stride', '1',
                  '--output', str(b_out)],
                 ['combine', '--a', str(output), '--b', str(b_out), '--confirm-s', '0.1', '--output', str(ab_out)]):
        result = subprocess.run([python, str(script)] + argv + (common if argv[0] != 'combine' else []),
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, (argv[0], result.stderr[-3000:])
        assert 'RESULT:' in result.stdout, (argv[0], result.stdout)
    episodes = json.loads((features / 'episodes.json').read_text())
    assert episodes['meta']['dim'] == 48 and episodes['meta']['frames'] == 4 * 5  # every frame of the four `first` episodes
    assert {ep for ep, entry in episodes['episodes'].items() if entry['split'] == 'val'} == {str(e) for e in val_episodes}
    index = np.load(features / 'index.npz')
    assert np.load(features / 'features.npy').shape == (20, 48) and sorted(set(index['frame'].tolist())) == [0, 1, 2, 3, 4]
    b_summary = json.loads((b_out / 'summary.json').read_text())
    assert b_summary['episodes'] == len(val_episodes) and len(b_summary['B_grid']) == 8 * 3
    assert {'tau', 'consecutive', 'false_stop_rate', 'median_abs_dt_s'} <= set(b_summary['methods']['B'])
    ab_summary = json.loads((ab_out / 'summary.json').read_text())
    assert set(ab_summary['methods']) == {'A', 'B', 'A+B'} and ab_summary['episodes'] == len(val_episodes)
    assert ab_summary['methods']['A']['done_rate'] == 1.0  # the gate at thresholds 0 proposes from the first step
    # A + B never fires before B alone does, and B alone never before the verifier's first confirmation
    for row in map(json.loads, (ab_out / 'episodes.jsonl').read_text().splitlines()):
        if row['A+B_done_step'] is not None:
            assert row['B_done_step'] is not None and row['A+B_done_step'] >= row['B_done_step']
