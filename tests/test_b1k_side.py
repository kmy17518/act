"""Side runs on the GB300 buildboxes: late goal fusion under DDP, --stop-file preemption, relative actions with
openpi pi05 quantile normalization, and validation L1 in action units."""

import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.multiprocessing as mp

from b1k_dataset import (ACTION_DELTA, CAMERAS, OBS_KEYS, B1KDataset, action_reference, decode_actions, extract_state,
                         guard_quantiles, normalize, preprocess_state, unnormalize)
from b1k_training import (committed_checkpoints, ddp_ignored_parameter_names, load_checkpoint, make_policy, parser,
                          resolve_val_split, train, unused_gradients)
from test_b1k import small_model_config, tiny_root, write_settle_windows  # noqa: F401  (fixture)


def late_config():
    """The buildbox2 recipe's late fusion (grid goal tokens through the shared backbone, zero goal tag, own goal LR),
    at test size, with a skipped decoder layer."""
    return dict(small_model_config(), state_dim=23, dec_layers=2, l1_normalization='valid', regime='image',
                task_onehot=False, goal_fusion='late', goal_views=[CAMERAS[0]], goal_tokens='grid',
                goal_encoder='shared_base', goal_tag_init='zero', goal_role_embedding=True, lr_goal=1e-4)


def late_batches(steps=3, per_rank=2, ranks=2):
    generator = torch.Generator().manual_seed(5)
    total = per_rank * ranks
    return [(torch.randn(total, 23, generator=generator), torch.rand(total, 3, 3, 32, 32, generator=generator),
             torch.randn(total, 4, 23, generator=generator), torch.zeros(total, 4, dtype=torch.bool),
             torch.rand(total, 1, 3, 32, 32, generator=generator)) for _ in range(steps)]


def late_ddp_worker(rank, world_size, store, result):
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.manual_seed(21)
    policy = make_policy(late_config(), 'cpu')
    optimizer = policy.configure_optimizers()
    zero_gradients = unused_gradients(policy)
    ignored = ddp_ignored_parameter_names(policy, zero_gradients)
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=rank, world_size=world_size)
    try:
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(policy, ignored)
        model = torch.nn.parallel.DistributedDataParallel(policy, broadcast_buffers=False, gradient_as_bucket_view=True)
        for step, (qpos, images, actions, pad, goal) in enumerate(late_batches()):
            part = slice(2 * rank, 2 * rank + 2)
            torch.manual_seed(100 * step + rank)  # the CVAE latent sample of this rank's half
            optimizer.zero_grad(set_to_none=True)
            model(qpos[part], images[part], actions[part], pad[part], goal=goal[part])['loss'].backward()
            for parameter, zeros in zero_gradients:
                parameter.grad = zeros
            optimizer.step()
    finally:
        dist.destroy_process_group()
    if rank == 0:
        torch.save({'ignored': ignored, 'state': policy.state_dict(),
                    'no_grad': [name for name, p in policy.named_parameters() if p.grad is None]}, result)


def test_late_fusion_trains_under_two_rank_ddp_like_one_process(tmp_path):
    """Late fusion's parameters all receive gradients except the padding head and the skipped decoder layers (the
    is_pad_head fix covers them), so two gloo ranks step without reducer errors and match one process that averages
    the two halves' gradients."""
    result = tmp_path / 'ddp.pt'
    mp.start_processes(late_ddp_worker, args=(2, str(tmp_path / 'store'), str(result)), nprocs=2, start_method='spawn')
    ddp = torch.load(result, weights_only=True)
    assert set(ddp['ignored']) == {name for name in ddp['state'] if name.startswith(
        ('model.is_pad_head.', 'model.transformer.decoder.layers.1.'))}
    assert set(ddp['no_grad']) == {'model.is_pad_head.weight', 'model.is_pad_head.bias'}
    torch.set_num_threads(1)
    torch.manual_seed(21)
    policy = make_policy(late_config(), 'cpu')
    optimizer = policy.configure_optimizers()
    assert [g['lr'] for g in optimizer.param_groups][-1] == 1e-4  # the goal group (goal tag)
    zero_gradients = unused_gradients(policy)
    for step, (qpos, images, actions, pad, goal) in enumerate(late_batches()):
        optimizer.zero_grad(set_to_none=True)
        for rank in range(2):
            part = slice(2 * rank, 2 * rank + 2)
            torch.manual_seed(100 * step + rank)
            (policy(qpos[part], images[part], actions[part], pad[part], goal=goal[part])['loss'] / 2).backward()
        for parameter, zeros in zero_gradients:
            parameter.grad = zeros
        optimizer.step()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(ddp['state'][name], value, atol=2e-6, rtol=1e-5, msg=name)
    assert ddp['state']['model.goal_role_embed.weight'].abs().sum() > 0  # the zero-initialized goal tag trains


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def train_rank_worker(rank, world_size, port, argvs, trigger, record):
    """One gloo rank of `train`; on rank 0, `trigger` = (stop file, check number) creates the stop file just before that
    stop check (check 1 runs before the first step, check k + 1 after step k). Writes this rank's stop checks."""
    import b1k_training
    os.environ.update(WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank), MASTER_ADDR='127.0.0.1',
                      MASTER_PORT=str(port))
    torch.set_num_threads(1)
    checks = []
    original = b1k_training.stop_requested

    def recorded(stop_file, device, distributed):
        if rank == 0 and trigger and len(checks) + 1 == trigger[1]:
            Path(trigger[0]).touch()
        checks.append(original(stop_file, device, distributed))
        return checks[-1]

    b1k_training.stop_requested = recorded
    returned = train(parser().parse_args(argvs[rank]))
    Path(record.format(rank=rank)).write_text(json.dumps({'checks': checks, 'returned': str(returned)}))


def run_ranks(tmp_path, argvs, trigger=None, name='run', timeout=600):
    """Two gloo ranks of `train`; ranks that disagree on when to stop deadlock in mismatched collectives and fail here."""
    record = str(tmp_path / f'{name}-rank{{rank}}.json')
    context = mp.start_processes(train_rank_worker, args=(2, free_port(), argvs, trigger, record), nprocs=2,
                                 start_method='spawn', join=False)
    deadline = time.monotonic() + timeout
    while not context.join(timeout=5):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            raise TimeoutError(f'{name}: the two ranks did not finish within {timeout} s')
    return [json.loads(Path(record.format(rank=rank)).read_text()) for rank in range(2)]


def test_stop_file_stops_both_gloo_ranks_at_the_same_step_and_leaves_a_resumable_checkpoint(tiny_root, tmp_path):
    output = tmp_path / 'run'
    stop = tmp_path / 'run.stop'  # rank 0's stop file; rank 1 watches a path that never exists
    common = ['--dataset-path', str(tiny_root), '--batch-size', '4', '--num-workers', '0', '--device', 'cpu',
              '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32', '--dim-feedforward', '64',
              '--enc-layers', '1', '--dec-layers', '2', '--nheads', '4', '--no-pretrained-backbone', '--dropout', '0',
              '--regime', 'image', '--goal-fusion', 'late', '--goal-tokens', 'grid', '--goal-encoder', 'shared_base',
              '--goal-tag-init', 'zero', '--lr-goal', '1e-4', '--save-every', '5', '--export-every', '5',
              '--save-total-limit', '3', '--output-dir', str(output)]
    argvs = [common + ['--max-steps', '10', '--stop-file', str(path)] for path in (stop, tmp_path / 'never.stop')]
    ranks = run_ranks(tmp_path, argvs, trigger=(str(stop), 4), name='stopped')  # created after step 3
    assert ranks[0]['checks'] == ranks[1]['checks'] == [False, False, False, True]
    assert ranks[0]['returned'] == str(output / 'step_00000003.pt')
    assert [p.name for p in committed_checkpoints(output)] == ['step_00000003.pt']
    assert [p.name for p in committed_checkpoints(output / 'export_queue/full')] == ['step_00000003.pt']
    assert (output / 'export_queue/full/step_00000003.pt').stat().st_ino == (output / 'step_00000003.pt').stat().st_ino
    assert not (output / 'export_queue/eval').exists()  # off the 5-step grid: no eval-only export
    stopped = load_checkpoint(output / 'step_00000003.pt')
    assert stopped['checkpoint_type'] == 'full' and stopped['step'] == 3 and stopped['optimizer']['state']
    assert stopped['train_config']['stop_file'] == str(stop) and stopped['train_config']['world_size'] == 2
    assert [json.loads(line)['step'] for line in (output / 'metrics.jsonl').read_text().splitlines()] == [1, 2, 3]

    # a stop file present at launch: no step, no checkpoint
    ranks = run_ranks(tmp_path, [a + ['--resume', str(output / 'step_00000003.pt')] for a in argvs], name='at-launch')
    assert ranks[0]['checks'] == ranks[1]['checks'] == [True] and ranks[0]['returned'] == 'None'
    assert [p.name for p in committed_checkpoints(output)] == ['step_00000003.pt']

    stop.unlink()
    ranks = run_ranks(tmp_path, [a + ['--resume', str(output / 'step_00000003.pt')] for a in argvs], name='resumed')
    assert ranks[0]['checks'] == ranks[1]['checks'] == [False] * 8
    assert ranks[0]['returned'] == str(output / 'step_00000010.pt')
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == list(range(1, 11))
    assert all(np.isfinite(record['loss']) for record in records)
    assert [p.name for p in committed_checkpoints(output)] == ['step_00000003.pt', 'step_00000005.pt',
                                                              'step_00000010.pt']
    assert [p.name for p in committed_checkpoints(output / 'export_queue/full')] == ['step_00000010.pt']
    assert [p.name for p in committed_checkpoints(output / 'export_queue/eval')] == ['step_00000005.pt',
                                                                                    'step_00000010.pt']
    resumed = load_checkpoint(output / 'step_00000010.pt')
    assert resumed['step'] == 10 and resumed['train_config']['resume'] == str(output / 'step_00000003.pt')
    assert resumed['model_config']['goal_fusion'] == 'late'


# --- relative actions, quantile normalization, action-unit validation L1 ---------------------------------------------

def add_episodes(root, count, shift=1.0):
    """`count` more episodes of task `first` (50, 51, ...), each with its own data file and actions shifted by
    `shift` per episode, on the camera segments of episode 42."""
    path = root / 'meta/episodes/chunk-042/file-000.parquet'
    for i in range(1, count + 1):
        episodes = pq.read_table(path).to_pylist()
        begin, episode_id = 200 + 10 * i, 49 + i
        episodes.append({**episodes[0], 'episode_index': episode_id, 'data/file_index': i, 'dataset_from_index': begin,
                         'dataset_to_index': begin + 5})
        pq.write_table(pa.Table.from_pylist(episodes), path)
        rows = [{**row, 'episode_index': episode_id, 'index': begin + row['frame_index'],
                 'action': [x + shift * i for x in row['action']]}
                for row in pq.read_table(root / 'data/chunk-042/file-000.parquet').to_pylist() if row['episode_index'] == 42]
        pq.write_table(pa.Table.from_pylist(rows), root / f'data/chunk-042/file-{i:03d}.parquet')


def brute_force_chunks(dataset, padded=False):
    """Every valid (with `padded`, every) chunk step's action relative to its frame's state, sample by sample."""
    values = []
    for ep in dataset.episodes:
        for frame in range(ep['length']):
            _, state, actions, is_pad, _ = dataset.raw_sample(ep['episode_index'], frame)
            steps = actions if padded else actions[~is_pad]
            values.append(steps.astype(np.float64) - action_reference(extract_state(state), ACTION_DELTA))
    return np.concatenate(values)


def realistic_rows(n, seed=0):
    """float32 R1Pro rows with the real ranges: base velocities, torso / arm joints in radians, finger positions in
    metres (each finger 0..0.05), gripper commands in [-1, 1]."""
    rng = np.random.default_rng(seed)
    state = rng.uniform(-3, 3, (n, 61)).astype(np.float32)
    state[:, [24, 25, 49, 50]] = rng.uniform(0, 0.05, (n, 4))
    actions = rng.uniform(-3, 3, (n, 100, 23)).astype(np.float32)
    actions[..., [14, 22]] = rng.uniform(-1, 1, (n, 100, 2))
    actions[..., :3] = rng.uniform(-0.5, 0.5, (n, 100, 3))
    return state, actions


def test_relative_actions_reproduce_the_dataset_actions_exactly_and_follow_openpi_mapping(tiny_root):
    state, actions = realistic_rows(64)
    proprio = extract_state(state)
    reference = action_reference(proprio, ACTION_DELTA)
    relative = actions.astype(np.float64) - reference[:, None]
    np.testing.assert_array_equal((relative + reference[:, None]).astype(np.float32), actions)  # all 23 dims, exactly
    np.testing.assert_array_equal(relative[..., :3], actions[..., :3])  # base velocity stays absolute
    joints = list(range(3, 14)) + list(range(15, 22))
    np.testing.assert_allclose(relative[..., joints], actions[..., joints] - proprio[:, None, joints], rtol=0, atol=1e-6)
    width = state[:, [24, 49]].astype(np.float64) + state[:, [25, 50]]  # q1 + q2 of each gripper
    np.testing.assert_allclose(relative[..., [14, 22]], actions[..., [14, 22]] - (20 * width[:, None] - 1), atol=1e-6)
    # the same through the dataset: decoding its normalized targets with the frame's state gives the recorded actions
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16))
    dataset.stats = dataset.compute_normalization_stats('quantile', 'relative')
    for ep in dataset.episodes:
        for frame in range(ep['length']):
            _, raw_state, raw_actions, is_pad, _ = dataset.raw_sample(ep['episode_index'], frame)
            sample = dataset.sample_at(ep['episode_index'], frame)
            decoded = decode_actions(sample[2].numpy().astype(np.float64), dataset.stats, extract_state(raw_state))
            np.testing.assert_allclose(decoded[~is_pad], raw_actions[~is_pad], rtol=0, atol=1e-5)  # float32 targets
            assert torch.equal(sample[3], torch.from_numpy(is_pad))  # padded steps stay masked
    dataset.close()


def test_quantile_normalization_round_trips_and_is_pi05s_formula():
    rng = np.random.default_rng(3)
    q01 = rng.normal(size=23)
    q99 = q01 + rng.uniform(0.05, 3, 23)
    stats = {'method': 'quantile', 'action_q01': q01.tolist(), 'action_q99': q99.tolist(),
             'qpos_q01': q01.tolist(), 'qpos_q99': q99.tolist()}
    values = rng.normal(size=(5, 100, 23)) * 2
    normalized = normalize(values, stats, 'action')
    np.testing.assert_allclose(normalized, (values - q01) / (q99 - q01 + 1e-6) * 2 - 1, rtol=1e-15)
    np.testing.assert_allclose(unnormalize(normalized, stats, 'action'), values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(normalize(np.stack([q01, q99]), stats, 'qpos'), [[-1] * 23, [1] * 23], atol=1e-4)
    # z-score keeps the exact arithmetic of the dataset and the server before this option
    zscore = {'action_mean': rng.normal(size=23).astype(np.float32).tolist(),
              'action_std': rng.uniform(0.01, 1, 23).astype(np.float32).tolist()}
    raw = rng.normal(size=(4, 23)).astype(np.float32)
    np.testing.assert_array_equal(normalize(raw, zscore, 'action'),
                                  (raw - np.asarray(zscore['action_mean'])) / np.asarray(zscore['action_std']))
    out = raw.copy()
    np.testing.assert_array_equal(unnormalize(out, zscore, 'action').astype(np.float32),
                                  (out * np.asarray(zscore['action_std'], dtype=np.float32)
                                   + np.asarray(zscore['action_mean'], dtype=np.float32)).astype(np.float32))
    narrow = guard_quantiles([0.0, 1.0], [0.001, 3.0], 0.02)
    np.testing.assert_allclose(narrow[0], [0.0005 - 0.01, 1.0])
    np.testing.assert_allclose(narrow[1], [0.0005 + 0.01, 3.0])
    assert narrow[2] == [0]


def test_quantile_statistics_cover_training_episodes_and_valid_chunk_steps_only(tiny_root, tmp_path):
    add_episodes(tiny_root, 9, shift=0.5)
    split = resolve_val_split(tiny_root, ['first'], None, 0.1, seed=0)
    train, val = split['train_episodes'], split['val_episodes']
    dataset = B1KDataset(tiny_root, ['first'], chunk_size=4, image_size=(16, 16), episodes=train)
    stats = dataset.compute_normalization_stats('quantile', 'relative')
    values = brute_force_chunks(dataset)
    assert stats['action_count'] == len(values) == sum(min(4, 5 - t) for _ in train for t in range(5))
    np.testing.assert_allclose(stats['action_q01_raw'], np.quantile(values, 0.01, axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(stats['action_q99_raw'], np.quantile(values, 0.99, axis=0), rtol=0, atol=1e-12)
    proprio = np.concatenate([extract_state(dataset.raw_sample(ep['episode_index'], t)[1])[None]
                              for ep in dataset.episodes for t in range(ep['length'])]).astype(np.float64)
    np.testing.assert_allclose(stats['qpos_q01_raw'], np.quantile(proprio, 0.01, axis=0), atol=1e-12)
    assert stats['count'] == len(proprio) == 5 * len(train) and not stats['approximate']
    narrow = [d for d in range(23) if stats['action_q99_raw'][d] - stats['action_q01_raw'][d] < 0.02]
    assert stats['guarded']['action'] == narrow
    assert all(stats['action_q99'][d] - stats['action_q01'][d] == pytest.approx(0.02) for d in narrow)
    # padded steps (zeros past the episode end) or the validation episode would change the quantiles
    assert not np.allclose(np.quantile(brute_force_chunks(dataset, padded=True), 0.01, axis=0), stats['action_q01_raw'])
    everything = B1KDataset(tiny_root, ['first'], chunk_size=4, image_size=(16, 16), episodes=train + val)
    assert everything.compute_normalization_stats('quantile', 'relative')['action_q99_raw'] != stats['action_q99_raw']
    # the trainer computes the same statistics from the training episodes of its split
    common = ['--dataset-path', str(tiny_root), '--task-names', 'first', '--batch-size', '2', '--num-workers', '0',
              '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
              '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
              '--no-pretrained-backbone', '--save-every', '2', '--val-fraction', '0.1', '--val-batches', '1',
              '--val-batch-size', '2', '--max-steps', '2', '--action-representation', 'relative',
              '--normalization', 'quantile', '--output-dir', str(tmp_path / 'run')]
    checkpoint = load_checkpoint(train_fn(common))
    saved = checkpoint['normalization']
    assert {key: value for key, value in saved.items() if key != 'fingerprint'} == \
        {key: value for key, value in stats.items() if key != 'fingerprint'}
    assert checkpoint['val_split']['val_episodes'] == val
    dataset.close()
    everything.close()


def train_fn(argv):
    return train(parser().parse_args(argv))


def tiny_common(root):
    return ['--dataset-path', str(root), '--batch-size', '2', '--num-workers', '0', '--device', 'cpu', '--chunk-size',
            '4', '--image-size', '16', '16', '--hidden-dim', '32', '--dim-feedforward', '64', '--enc-layers', '1',
            '--dec-layers', '1', '--nheads', '4', '--no-pretrained-backbone', '--save-every', '2']


RELQ = ['--action-representation', 'relative', '--normalization', 'quantile']


def test_server_returns_absolute_actions_for_a_relative_quantile_checkpoint(tiny_root, tmp_path):
    from b1k_server import PolicyPredictor, Session
    from test_b1k import observation
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(train_fn(tiny_common(tiny_root) + RELQ + ['--max-steps', '2', '--output-dir',
                                                                           str(tmp_path / 'relq')]))
    assert checkpoint['model_config']['action_representation'] == 'relative'
    assert checkpoint['adapter_config']['action_transform'] == 'relative'  # code before the option refuses it
    stats = checkpoint['normalization']
    assert stats['method'] == 'quantile' and stats['action_delta'] == ACTION_DELTA
    predictor = PolicyPredictor(checkpoint, 'cpu')
    first, second = observation(), observation()
    first['robot_r1::proprio'], second['robot_r1::proprio'] = realistic_rows(1, 1)[0], realistic_rows(1, 2)[0]
    session = Session(predictor, checkpoint)

    def inputs(obs):
        qpos = torch.stack([preprocess_state(s, 7, stats, checkpoint['task_map'], session.task_onehot,
                                             gripper_state='sum') for s in obs['robot_r1::proprio']])
        return qpos, session.batch_images(obs, OBS_KEYS, 1)

    def manual(obs):  # the network's normalized chunk, unnormalized, plus the reference of this observation's state
        with torch.inference_mode():
            raw = predictor.policy(*inputs(obs)).numpy()
        return (unnormalize(raw, stats, 'action')
                + action_reference(extract_state(obs['robot_r1::proprio']), ACTION_DELTA)[:, None]).astype(np.float32)

    chunks = [manual(first), manual(second)]
    np.testing.assert_allclose(predictor(*inputs(first), states=first['robot_r1::proprio']), chunks[0], rtol=1e-6,
                               atol=1e-6)
    with pytest.raises(ValueError, match='decode only with the state'):
        predictor(*inputs(first))
    weights = np.exp(-0.01 * np.arange(2))
    for snap in (None, 0.9):
        chunked = Session(predictor, checkpoint, action_horizon=4, gripper_snap=snap)
        aggregated = Session(predictor, checkpoint, temporal_agg=True, gripper_snap=snap)
        actual = [chunked.act(first), chunked.act(second), aggregated.act(first), aggregated.act(second)]
        # chunked: the second action is the first plan's, decoded with the first state; aggregation: each chunk
        # decoded with its own observation's state, then averaged
        expected = [chunks[0][:, 0], chunks[0][:, 1], chunks[0][:, 0],
                    (chunks[0][:, 1] * weights[0] + chunks[1][:, 0] * weights[1]) / weights.sum()]
        for got, want in zip(actual, expected):
            if snap is not None:
                grippers = np.clip(want[:, [14, 22]], -1, 1)
                want = want.copy()
                want[:, [14, 22]] = np.where(grippers > snap, 1.0, grippers)
            np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_checkpoints_without_the_new_keys_load_as_absolute_zscore_and_resume_checks_conflicts(tiny_root, tmp_path):
    from b1k_server import PolicyPredictor, Session
    from test_b1k import observation
    torch.set_num_threads(1)
    common = tiny_common(tiny_root)
    path = train_fn(common + ['--max-steps', '2', '--output-dir', str(tmp_path / 'abs')])
    new = load_checkpoint(path)
    assert (new['model_config']['action_representation'], new['model_config']['normalization']) == ('absolute', 'zscore')
    assert new['adapter_config']['action_transform'] == 'identity' and 'method' not in new['normalization']
    old = torch.load(path, weights_only=True)
    for key in ('action_representation', 'normalization'):
        del old['model_config'][key]
    legacy = tmp_path / 'legacy.pt'
    torch.save(old, legacy)
    for checkpoint in (new, load_checkpoint(legacy)):  # served exactly like before the option
        stats = checkpoint['normalization']
        session = Session(PolicyPredictor(checkpoint, 'cpu'), checkpoint, action_horizon=2)
        qpos = preprocess_state(observation()['robot_r1::proprio'][0], 7, stats, checkpoint['task_map'],
                                gripper_state='sum')
        with torch.inference_mode():
            raw = session.predictor.policy(qpos[None], session.batch_images(observation(), OBS_KEYS, 1)).numpy()
        want = (raw * np.asarray(stats['action_std'], dtype=np.float32)
                + np.asarray(stats['action_mean'], dtype=np.float32)).astype(np.float32)[:, 0]
        np.testing.assert_array_equal(session.act(observation()), want)
    resumed = load_checkpoint(train_fn(common + ['--max-steps', '3', '--output-dir', str(tmp_path / 'abs'),
                                                 '--resume', str(legacy)]))
    assert 'action_representation' not in resumed['model_config'] and 'method' not in resumed['normalization']
    for flag, value in (('--action-representation', 'relative'), ('--normalization', 'quantile')):
        with pytest.raises(ValueError, match=f'{flag} differs from checkpoint'):
            train_fn(common + ['--max-steps', '4', '--output-dir', str(tmp_path / 'conflict'), flag, value,
                               '--resume', str(legacy)])
    relq = train_fn(common + RELQ + ['--max-steps', '2', '--output-dir', str(tmp_path / 'relq')])
    with pytest.raises(ValueError, match='--action-representation differs from checkpoint'):
        train_fn(common + ['--max-steps', '3', '--output-dir', str(tmp_path / 'conflict'), '--action-representation',
                           'absolute', '--resume', str(relq)])
    again = load_checkpoint(train_fn(common + ['--max-steps', '3', '--output-dir', str(tmp_path / 'relq'),
                                               '--resume', str(relq)]))
    assert (again['model_config']['action_representation'], again['model_config']['normalization']) == \
        ('relative', 'quantile')
    assert again['normalization'] == load_checkpoint(relq)['normalization'] and again['step'] == 3
    with pytest.raises(ValueError, match='--action-representation relative needs --gripper-state sum'):
        train_fn(common + RELQ + ['--max-steps', '2', '--output-dir', str(tmp_path / 'fingers'), '--gripper-state',
                                  'fingers'])


def test_validation_l1_in_action_units_compares_absolute_actions_of_any_representation(tiny_root, tmp_path):
    from contextlib import nullcontext
    from b1k_training import evaluate_validation
    torch.set_num_threads(1)
    for normalization, representation in (('zscore', 'absolute'), ('quantile', 'relative'), ('quantile', 'absolute')):
        dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(32, 32), task_onehot=False)
        dataset.stats = dataset.compute_normalization_stats(normalization, representation)
        torch.manual_seed(14)
        policy = make_policy(dict(small_model_config(), state_dim=23, l1_normalization='valid'), 'cpu')
        metrics = evaluate_validation(policy, dataset, 2, 3, None, dataset.task_map, torch.device('cpu'), nullcontext,
                                      False, 0)
        # independently: zero-latent predictions decoded with each sample's raw state vs the recorded actions
        error, count = 0.0, 0
        policy.eval()
        from b1k_dataset import StepBatchSampler
        for batch in StepBatchSampler(dataset, 3, 0, 2, 0):
            for index in batch:
                position = int(np.searchsorted(dataset.ends, index, side='right'))
                ep, frame = dataset.episodes[position], int(index - dataset.starts[position])
                images, qpos, _, is_pad, _, _ = dataset.sample_at(ep['episode_index'], frame)
                _, state, actions, _, _ = dataset.raw_sample(ep['episode_index'], frame)
                with torch.no_grad():
                    prediction = policy(qpos[None], images[None]).numpy().astype(np.float64)
                decoded = decode_actions(prediction, dataset.stats, extract_state(state)[None])[0]
                valid = ~is_pad.numpy()
                error += np.abs(decoded[valid] - actions[valid]).sum()
                count += int(valid.sum()) * 23
        assert metrics['val/l1_action_units'] == pytest.approx(error / count, rel=1e-5), (normalization, representation)
        families = {key.split('/')[-1] for key in metrics if key.startswith('val/l1/')}
        assert families and {key for key in metrics if key.startswith('val/l1_action_units/')} == \
            {f'val/l1_action_units/{family}' for family in families}
        dataset.close()


def test_smoke_client_passes_against_a_served_relative_late_fusion_checkpoint(tiny_root, tmp_path, capsys):
    import asyncio
    import importlib.util
    import sys
    import threading
    from b1k_server import B1KServer, PolicyPredictor
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(train_fn(tiny_common(tiny_root) + RELQ + [
        '--max-steps', '2', '--output-dir', str(tmp_path / 'relq-late'), '--regime', 'image', '--goal-fusion', 'late']))
    port = free_port()
    server = B1KServer(checkpoint, PolicyPredictor(checkpoint, 'cpu'), '127.0.0.1', port, action_horizon=3)
    threading.Thread(target=lambda: asyncio.run(server.run()), daemon=True).start()
    spec = importlib.util.spec_from_file_location('smoke_client', Path(__file__).parents[1] / 'scripts/b1k/smoke_client.py')
    client = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)
    for _ in range(100):  # until the server listens
        try:
            socket.create_connection(('127.0.0.1', port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.1)
    sys_argv = sys.argv
    sys.argv = ['smoke_client.py', '--dataset-path', str(tiny_root), '--task-names', 'first', 'second', '--port', str(port)]
    try:
        client.main()
    finally:
        sys.argv = sys_argv
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result['result'] == 'PASS' and 'plan_decoded_at_plan_time' in result['checks']
    assert result['metadata']['action_representation'] == 'relative' and result['metadata']['served_actions'] == 'absolute'

def test_offline_validation_on_another_settle_cut(tiny_root, tmp_path, capsys):
    """val_action_units.py --settle-steps validates a checkpoint on another settle cut of its run's validation episodes:
    the run's own by default (the logged metrics), the programs alone with 0."""
    import importlib.util
    import sys
    torch.set_num_threads(1)
    add_episodes(tiny_root, 9, shift=0.5)
    write_settle_windows(tiny_root, {42: 3, 99: 2, **{50 + i: 3 for i in range(9)}})  # 5 recorded frames each
    output = tmp_path / 'run'
    train_fn(tiny_common(tiny_root) + ['--task-names', 'first', '--max-steps', '2', '--output-dir', str(output),
                                       '--settle-steps', '1', '--val-fraction', '0.1', '--val-every', '2',
                                       '--val-batches', '2', '--val-batch-size', '3', '--export-every', '2'])
    logged = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()][-1]
    spec = importlib.util.spec_from_file_location('val_action_units',
                                                  Path(__file__).parents[1] / 'scripts/b1k/val_action_units.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    def evaluate(*extra):
        sys_argv, sys.argv = sys.argv, ['val_action_units.py', '--checkpoint',
                                        str(output / 'export_queue/eval/step_00000002.pt'), '--device', 'cpu', *extra]
        try:
            script.main()
        finally:
            sys.argv = sys_argv
        return json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    own, respelled, programs = evaluate(), evaluate('--settle-steps', '01'), evaluate('--settle-steps', '0')
    assert own['settle_steps'] == own['run_settle_steps'] == '1' and own['val_frames'] == 4  # 1 val episode: 3 + 1
    for key in ('val/l1', 'val/l1_action_units', 'val/inference_l1'):
        assert own['metrics'][key] == pytest.approx(logged[key], rel=1e-6)
    measured = lambda result: {k: v for k, v in result['metrics'].items() if not k.startswith('timing/')}
    assert respelled['settle_steps'] == '1' and measured(respelled) == measured(own)
    assert programs['settle_steps'] == '0' and programs['run_settle_steps'] == '1' and programs['val_frames'] == 3
    assert programs['metrics']['val/l1'] != own['metrics']['val/l1']


@pytest.mark.parametrize('mode', [[], RELQ])
def test_offline_action_unit_l1_matches_the_logged_validation_metric(tiny_root, tmp_path, mode, capsys):
    import importlib.util
    import sys
    torch.set_num_threads(1)
    add_episodes(tiny_root, 9, shift=0.5)
    output = tmp_path / 'run'
    train_fn(tiny_common(tiny_root) + mode + ['--task-names', 'first', '--max-steps', '2', '--output-dir', str(output),
                                              '--val-fraction', '0.1', '--val-every', '2', '--val-batches', '2',
                                              '--val-batch-size', '3', '--export-every', '2'])
    logged = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()][-1]
    checkpoint = output / 'export_queue/eval/step_00000002.pt'
    if not mode:  # a checkpoint from before the option: no action representation / normalization keys
        old = torch.load(checkpoint, weights_only=True)
        del old['model_config']['action_representation'], old['model_config']['normalization']
        checkpoint = tmp_path / 'old_eval.pt'
        torch.save(old, checkpoint)
    spec = importlib.util.spec_from_file_location('val_action_units',
                                                  Path(__file__).parents[1] / 'scripts/b1k/val_action_units.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    sys_argv, sys.argv = sys.argv, ['val_action_units.py', '--checkpoint', str(checkpoint), '--device', 'cpu']
    try:
        script.main()
    finally:
        sys.argv = sys_argv
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result['step'] == 2 and result['action_representation'] == ('relative' if mode else 'absolute')
    for key in ('val/l1_action_units', 'val/inference_l1'):
        assert result['metrics'][key] == pytest.approx(logged[key], rel=1e-6)
