"""Boundary, normalization, upstream numerical, and BEHAVIOR protocol regressions."""

import asyncio
import json
import msgpack
from unittest.mock import patch

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import websockets.asyncio.client
import websockets.asyncio.server

from b1k_dataset import (B1KDataset, OBS_KEYS, STATE_INDICES, VIDEO_KEYS, StepBatchSampler,
                         preprocess_image, preprocess_state)
from b1k_server import B1KServer, Session, packb, unpackb
from b1k_training import load_checkpoint, make_policy, parser, train
from detr.main import get_args_parser
from detr.models.detr_vae import CNNMLP, DETRVAE
from policy import CNNMLPPolicy
from utils import EpisodicDataset, get_norm_stats


@pytest.fixture
def tiny_root(tmp_path):
    root = tmp_path / 'lerobot'
    (root / 'meta/episodes/chunk-042').mkdir(parents=True)
    (root / 'data/chunk-042').mkdir(parents=True)
    info = {'codebase_version': 'v3.0', 'fps': 10,
            'features': {'action': {'shape': [23]}, 'observation.state': {'shape': [61]}},
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mkv'}
    (root / 'meta/info.json').write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist([{'task_index': 7, 'task': 'first'},
                                        {'task_index': 9, 'task': 'second'},
                                        {'task_index': 11, 'task': 'unavailable'}]), root / 'meta/tasks.parquet')
    for c, key in enumerate(VIDEO_KEYS):
        video = root / info['video_path'].format(video_key=key, chunk_index=42, file_index=0)
        video.parent.mkdir(parents=True)
        with av.open(str(video), 'w') as container:
            stream = container.add_stream('ffv1', rate=10)
            stream.width = stream.height = 16
            stream.pix_fmt = 'bgr0'
            for i in range(20):
                image = np.full((16, 16, 3), i * 10 + c, np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    rows, episodes = [], []
    for e, (episode_id, task, begin) in enumerate([(42, 7, 100), (99, 9, 900)]):
        ep = {'episode_index': episode_id, 'task_index': task, 'length': 5,
              'dataset_from_index': begin, 'dataset_to_index': begin + 5,
              'data/chunk_index': 42, 'data/file_index': 0}
        for c, key in enumerate(VIDEO_KEYS):
            offset = (e * 8 + c * 2) / 10
            ep.update({f'videos/{key}/chunk_index': 42, f'videos/{key}/file_index': 0,
                       f'videos/{key}/from_timestamp': offset, f'videos/{key}/to_timestamp': offset + 0.6})
        episodes.append(ep)
        for frame, timestamp in enumerate([0., .2, .3, .4, .5]):
            state = np.arange(61, dtype=np.float32) * .01 + frame + e
            action = np.arange(23, dtype=np.float32) * .1 + frame - e
            state[0] = 0
            action[0] = 0
            rows.append({'episode_index': episode_id, 'task_index': task, 'index': begin + frame,
                         'frame_index': frame, 'timestamp': timestamp,
                         'observation.state': state.tolist(), 'action': action.tolist()})
    pq.write_table(pa.Table.from_pylist(rows), root / 'data/chunk-042/file-000.parquet', row_group_size=3)
    pq.write_table(pa.Table.from_pylist(episodes), root / 'meta/episodes/chunk-042/file-000.parquet')
    return root


def hdf5_reference(root, tmp_path, stats, sim=True):
    dataset = B1KDataset(root, chunk_size=4, image_size=(16, 16))
    directory = tmp_path / 'hdf5'
    directory.mkdir(exist_ok=True)
    for i, ep in enumerate(dataset.episodes):
        samples = [dataset.raw_sample(ep['episode_index'], frame) for frame in range(ep['length'])]
        with h5py.File(directory / f'episode_{i}.hdf5', 'w') as file:
            file.attrs['sim'] = sim
            file['observations/qpos'] = np.stack([sample[1][STATE_INDICES] for sample in samples])
            file['observations/qvel'] = np.zeros((ep['length'], 25), dtype=np.float32)
            file['action'] = np.stack([sample[2][0] for sample in samples])
            for c, key in enumerate(VIDEO_KEYS):
                file[f'observations/images/{key}'] = np.stack([sample[0][c] for sample in samples])
    dataset.close()
    return directory


def test_exact_hdf5_statistics_and_sample_boundaries(tiny_root, tmp_path):
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16))
    stats = dataset.compute_stats()
    dataset.stats = stats
    directory = hdf5_reference(tiny_root, tmp_path, stats)
    reference_stats = get_norm_stats(directory, 2)
    for key in ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']:
        np.testing.assert_allclose(stats[key], reference_stats[key], atol=5e-7, rtol=2e-6)
    assert stats['qpos_std'][0] == pytest.approx(.01)
    assert stats['action_std'][0] == pytest.approx(.01)
    assert stats['count'] == 10 and not stats['approximate'] and stats['std_correction'] == 1
    for e, ep in enumerate(dataset.episodes):
        for frame in [0, 1, 3, 4]:
            with patch('numpy.random.choice', return_value=frame):
                reference = EpisodicDataset([e], str(directory), VIDEO_KEYS, reference_stats)[0]
            sample = dataset.sample_at(ep['episode_index'], frame)
            torch.testing.assert_close(sample[0], reference[0], atol=0, rtol=0)
            torch.testing.assert_close(sample[1][:25], reference[1], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(sample[2], reference[2][:4], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(sample[3], reference[3][:4])
            assert sample[1][25 + e] == 1
    assert dataset.sample_at(42, 4)[3].tolist() == [False, True, True, True]
    assert len(dataset._groups) <= 2 and len(dataset._videos) <= 3
    dataset.close()


def test_real_hdf5_shift_is_deliberately_absent(tiny_root, tmp_path):
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16))
    dataset.stats = dataset.compute_stats()
    directory = hdf5_reference(tiny_root, tmp_path, dataset.stats, sim=False)
    reference_stats = {key: np.asarray(dataset.stats[key], dtype=np.float32)
                       for key in ['qpos_mean', 'qpos_std', 'action_mean', 'action_std']}
    with patch('numpy.random.choice', return_value=4):
        reference = EpisodicDataset([0], str(directory), VIDEO_KEYS, reference_stats)[0]
    aligned = dataset.sample_at(42, 4)
    assert reference[3][:4].tolist() == [False, False, True, True]
    assert aligned[3].tolist() == [False, True, True, True]
    torch.testing.assert_close(aligned[2][0], reference[2][1])
    assert not torch.equal(aligned[2][0], reference[2][0])
    dataset.close()


def test_timestamps_camera_offsets_and_read_only(tiny_root):
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in tiny_root.rglob('*') if p.is_file()}
    dataset = B1KDataset(tiny_root, chunk_size=4)
    assert not dataset._groups and not dataset._videos
    images, state, actions, pad, task = dataset.raw_sample(99, 1)
    # Timestamp .2, not frame_index/fps=.1; each camera has a different offset.
    assert [int(image[0, 0, 0]) for image in images] == [100, 121, 142]
    assert task == 9 and not pad.any()
    dataset.close()
    after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in tiny_root.rglob('*') if p.is_file()}
    assert before == after


def test_partial_noncontiguous_ids_task_filters(tiny_root):
    with pytest.raises(ValueError, match='Unknown task'):
        B1KDataset(tiny_root, ['typo'])
    with pytest.raises(ValueError, match='No complete local episodes'):
        B1KDataset(tiny_root, ['unavailable'])
    with pytest.raises(ValueError, match='No complete local episodes'):
        B1KDataset(tiny_root, ['first', 'unavailable'])
    dataset = B1KDataset(tiny_root, ['second'])
    assert dataset.task_map == {9: 'second'}
    assert list(dataset.by_id) == [99] and len(dataset) == 5
    dataset.stats = dataset.compute_stats()
    assert dataset.stats['count'] == 5
    assert dataset[0][1].shape == (26,)
    assert dataset[4][3][1:].all()
    metadata_path = tiny_root / 'meta/episodes/chunk-042/file-000.parquet'
    pq.write_table(pa.Table.from_pylist([dataset.episodes[0]]), metadata_path)
    assert B1KDataset(tiny_root).task_map == {9: 'second'}
    with pytest.raises(ValueError, match='No complete local episodes'):
        B1KDataset(tiny_root, ['first'])
    with pytest.raises(IndexError):
        dataset.sample_at(99, 5)
    dataset.close()


def test_episode_task_name_fallback(tiny_root):
    path = tiny_root / 'meta/episodes/chunk-042/file-000.parquet'
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row['tasks'] = [{7: 'first', 9: 'second'}[row.pop('task_index')]]
    pq.write_table(pa.Table.from_pylist(rows), path)
    dataset = B1KDataset(tiny_root, ['second'])
    assert dataset.task_map == {9: 'second'} and list(dataset.by_id) == [99]


def test_corrupt_rows_rejected(tiny_root):
    path = tiny_root / 'data/chunk-042/file-000.parquet'
    rows = pq.read_table(path).to_pylist()
    rows[4]['episode_index'] = 99
    pq.write_table(pa.Table.from_pylist(rows), path)
    dataset = B1KDataset(tiny_root)
    with pytest.raises(ValueError, match='Corrupt episode'):
        dataset.raw_sample(42, 4)


def test_step_sampler_resume_and_no_frame_permutation(tiny_root):
    dataset = B1KDataset(tiny_root)
    whole = list(StepBatchSampler(dataset, 3, 0, 5, 17))
    resumed = list(StepBatchSampler(dataset, 3, 2, 5, 17))
    assert whole[2:] == resumed
    assert all(0 <= i < len(dataset) for batch in whole for i in batch)


def small_model_config():
    return dict(state_dim=27, action_dim=23, num_queries=4, hidden_dim=32, dim_feedforward=64,
                enc_layers=1, dec_layers=1, nheads=4, kl_weight=10, lr=1e-4, lr_backbone=1e-5,
                weight_decay=1e-4, pretrained_backbone=False, camera_names=VIDEO_KEYS, dropout=0.0)


def test_real_act_numerical_loss_backward_and_padding():
    torch.set_num_threads(1)
    torch.manual_seed(10)
    policy = make_policy(small_model_config(), 'cpu')
    assert isinstance(policy.model, DETRVAE)
    qpos = torch.randn(2, 27)
    images = torch.rand(2, 3, 3, 32, 32)
    actions = torch.randn(2, 4, 23)
    pad = torch.tensor([[False, False, True, True], [False, False, False, True]])
    rng = torch.get_rng_state()
    losses = policy(qpos, images, actions, pad)
    torch.set_rng_state(rng)
    normalized_images = (images - torch.tensor([.485, .456, .406])[None, None, :, None, None])
    normalized_images /= torch.tensor([.229, .224, .225])[None, None, :, None, None]
    prediction, _, (mu, logvar) = policy.model(qpos, normalized_images, None, actions, pad)
    l1 = ((prediction - actions).abs() * ~pad[..., None]).mean()
    kl = (-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
    torch.testing.assert_close(losses['l1'], l1)
    torch.testing.assert_close(losses['kl'], kl)
    torch.testing.assert_close(losses['loss'], l1 + 10 * kl)
    changed = actions.clone()
    changed[pad] += 100
    torch.set_rng_state(rng)
    altered = policy(qpos, images, changed, pad)
    torch.testing.assert_close(losses['loss'], altered['loss'])
    before = policy.model.action_head.weight.detach().clone()
    optimizer = policy.configure_optimizers()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        values = policy(qpos, images, actions, pad)
        values['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        optimizer.step()
    assert not torch.equal(before, policy.model.action_head.weight)
    policy.eval()
    assert policy(qpos, images).shape == (2, 4, 23)
    defaults = get_args_parser()
    assert defaults.get_default('state_dim') == defaults.get_default('action_dim') == 14
    assert defaults.get_default('hidden_dim') == 256
    assert parser().get_default('hidden_dim') == 512


@pytest.mark.parametrize('policy_class', ['ACT', 'CNNMLP'])
def test_checkpoint_resume_matches_uninterrupted_and_serves_without_dataset(tiny_root, tmp_path, policy_class):
    common = ['--dataset-path', str(tiny_root), '--batch-size', '2', '--num-workers', '0',
              '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
              '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
              '--no-pretrained-backbone', '--save-every', '1', '--dropout', '0.1', '--policy-class', policy_class]
    if policy_class == 'CNNMLP':
        common += ['--image-size', '416', '416', '--batch-size', '1']
    full = tmp_path / 'full'
    split = tmp_path / 'split'
    train(parser().parse_args(common + ['--output-dir', str(full), '--max-steps', '3']))
    train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '2']))
    resumed_path = train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '3',
                           '--resume', str(split / 'step_00000002.pt')]))
    resumed = load_checkpoint(resumed_path)
    expected = load_checkpoint(full / 'step_00000003.pt')
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)
    assert resumed['step'] == 3 and resumed['optimizer']['state']
    with pytest.raises(FileExistsError):
        train(parser().parse_args(common + ['--output-dir', str(full), '--max-steps', '3']))
    tiny_root.rename(tmp_path / 'dataset_hidden')
    from b1k_server import ACTPredictor
    predictor = ACTPredictor(resumed, 'cpu')
    session = Session(predictor, resumed, 2 if policy_class == 'ACT' else 1)
    assert session.act(observation()).shape == (1, 23)
    assert resumed['model_config']['policy_class'] == policy_class
    if policy_class == 'CNNMLP':
        assert isinstance(predictor.policy, CNNMLPPolicy)
        assert resumed['model_config']['num_queries'] == 1
    else:
        del resumed['model_config']['policy_class']
        legacy_path = tmp_path / 'old_b1k.pt'
        torch.save(resumed, legacy_path)
        legacy = load_checkpoint(legacy_path)
        old_predictor = ACTPredictor(legacy, 'cpu')
        old_action = Session(old_predictor, legacy, 2).act(observation())
        np.testing.assert_array_equal(old_action, Session(predictor, expected, 2).act(observation()))


@pytest.mark.parametrize('position', ['sine', 'learned'])
@pytest.mark.parametrize('pre_norm', [False, True])
def test_act_architecture_variants_batch_two(position, pre_norm):
    torch.set_num_threads(1)
    config = dict(small_model_config(), policy_class='ACT', position_embedding=position, pre_norm=pre_norm)
    policy = make_policy(config, 'cpu')
    qpos, images = torch.rand(2, 27), torch.rand(2, 3, 3, 32, 32)
    actions, pad = torch.rand(2, 4, 23), torch.zeros(2, 4, dtype=torch.bool)
    loss = policy(qpos, images, actions, pad)['loss']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    assert policy.model.encoder.layers[0].normalize_before == pre_norm
    assert policy.model.transformer.decoder.layers[0].normalize_before == pre_norm
    if position == 'learned':
        assert policy.model.backbones[0][1].row_embed.weight.grad is not None
    policy.eval()
    assert policy(qpos, images).shape == (2, 4, 23)


@pytest.mark.parametrize('image_size', [(480, 640), (416, 416)])
def test_cnnmlp_real_model_spatial_shape_first_action_mse(image_size):
    torch.set_num_threads(1)
    config = dict(small_model_config(), policy_class='CNNMLP', num_queries=1, image_size=image_size)
    policy = make_policy(config, 'cpu')
    assert isinstance(policy, CNNMLPPolicy) and isinstance(policy.model, CNNMLP)
    qpos, images = torch.rand(1, 27), torch.rand(1, 3, 3, *image_size)
    actions = torch.rand(1, 4, 23)
    losses = policy(qpos, images, actions)
    normalized = (images - torch.tensor([.485, .456, .406])[None, None, :, None, None])
    normalized /= torch.tensor([.229, .224, .225])[None, None, :, None, None]
    with torch.no_grad():
        prediction = policy.model(qpos, normalized, None)
        expected = torch.nn.functional.mse_loss(prediction, actions[:, 0])
    torch.testing.assert_close(losses['mse'], expected)
    losses['loss'].backward()
    assert prediction.shape == (1, 23)
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    assert policy.model.mlp[-1].weight.grad.abs().sum() > 0
    for backbone in policy.model.backbones:
        assert backbone[0].body.conv1.weight.grad.abs().sum() > 0
    features = 32 * ((image_size[0] + 31) // 32 - 12) * ((image_size[1] + 31) // 32 - 12)
    assert policy.model.mlp[0].in_features == features * 3 + 27
    assert policy.model.mlp[-1].out_features == 23
    assert policy.model.action_head.weight.grad is None


def test_cnnmlp_original_dimensions_and_checkpoint_keys():
    config = dict(small_model_config(), policy_class='CNNMLP', num_queries=1)
    config.pop('state_dim')
    config.pop('action_dim')
    config['camera_names'] = ['head']
    policy = make_policy(config, 'cpu')
    assert policy.model.image_size == (480, 640)
    assert policy.model.mlp[0].weight.shape == (1024, 768 + 14)
    assert policy.model.mlp[-1].weight.shape == (14, 1024)
    assert policy.model.action_head.weight.shape == (14, 1000)
    import subprocess
    import types
    source = subprocess.check_output(['git', 'show', 'HEAD:detr/models/detr_vae.py'], text=True)
    upstream = types.ModuleType('detr.models._legacy_cnnmlp_reference')
    upstream.__package__ = 'detr.models'
    exec(compile(source, 'HEAD:detr/models/detr_vae.py', 'exec'), upstream.__dict__)
    reference = upstream.CNNMLP(list(policy.model.backbones), 14, ['head'])
    reference.load_state_dict(policy.model.state_dict(), strict=True)
    assert list(reference.state_dict()) == list(policy.model.state_dict())
    images = torch.rand(1, 1, 3, 480, 640)
    qpos = torch.rand(1, 14)
    torch.testing.assert_close(policy.model(qpos, images, None), reference(qpos, images, None), atol=0, rtol=0)
    restored = make_policy(config, 'cpu', restoring=True)
    restored.model.load_state_dict(reference.state_dict(), strict=True)
    with torch.no_grad():
        assert restored(torch.zeros(1, 14), torch.zeros(1, 1, 3, 480, 640)).shape == (1, 14)
    with pytest.raises(ValueError, match='13x13'):
        make_policy(dict(config, image_size=[240, 240]), 'cpu')
    with pytest.raises(ValueError, match='expects image size'):
        restored(torch.zeros(1, 14), torch.zeros(1, 1, 3, 416, 416))


def test_policy_selection_rejects_invalid_modes_and_preserves_act_default():
    from b1k_training import policy_class
    assert policy_class({}) == 'ACT'
    with pytest.raises(ValueError, match='Unsupported policy class'):
        make_policy(dict(small_model_config(), policy_class='other'), 'cpu')
    with pytest.raises(ValueError, match='one action'):
        make_policy(dict(small_model_config(), policy_class='CNNMLP'), 'cpu')
    checkpoint = checkpoint_stub()
    checkpoint['model_config'] = {'policy_class': 'CNNMLP', 'num_queries': 1}
    predictor = lambda qpos, images: np.zeros((len(qpos), 23), dtype=np.float32)
    server = B1KServer(checkpoint, predictor)
    assert server.metadata['policy'] == 'CNNMLP'
    assert server.metadata['execution_mode'] == 'single_action'
    assert server.metadata['action_horizon'] == server.metadata['chunk_size'] == 1
    assert Session(predictor, checkpoint).act(observation()).shape == (1, 23)
    with pytest.raises(ValueError, match='only supported for ACT'):
        Session(predictor, checkpoint, temporal_agg=True)
    with pytest.raises(ValueError, match='between 1'):
        Session(predictor, checkpoint, action_horizon=2)
    with pytest.raises(ValueError, match='queries every observation'):
        Session(predictor, checkpoint_stub(), action_horizon=2, temporal_agg=True)


def checkpoint_stub():
    return {'step': 2, 'task_map': {7: 'first', 9: 'second'},
            'normalization': {'qpos_mean': [0.] * 25, 'qpos_std': [1.] * 25,
                              'action_mean': [0.] * 23, 'action_std': [1.] * 23},
            'adapter_config': {'image_size': [16, 16]}, 'model_config': {'num_queries': 4}}


class CountingPredictor:
    def __init__(self):
        self.calls = []

    def __call__(self, qpos, images):
        self.calls.append((qpos.clone(), images.clone()))
        base = qpos[:, 1].numpy()[:, None, None]
        return np.broadcast_to(base + np.arange(4, dtype=np.float32)[None, :, None], (len(qpos), 4, 23)).copy()


def observation(batch=1, task=7, value=10, rgba=False):
    return {'robot_r1::proprio': np.full((batch, 61), value, dtype=np.float32), 'task_id': task,
            **{key: np.full((batch, 16, 16, 4 if rgba else 3), 128, dtype=np.uint8) for key in OBS_KEYS}}


def test_preprocessing_session_replan_batch_tasks_and_transaction():
    predictor = CountingPredictor()
    session = Session(predictor, checkpoint_stub(), 2)
    obs = observation(2, np.array([7, 9]), rgba=True)
    first = session.act(obs)
    assert first.shape == (2, 23) and first.dtype == np.float32
    qpos, images = predictor.calls[-1]
    torch.testing.assert_close(qpos[0], preprocess_state(obs['robot_r1::proprio'][0], 7,
                                                       session.stats, session.task_map))
    torch.testing.assert_close(images[0, 0], preprocess_image(obs[OBS_KEYS[0]][0], (16, 16)))
    invalid = observation(2, 777)
    old_positions = session.positions.copy()
    with pytest.raises(ValueError, match='Unseen task_id'):
        session.act(invalid)
    np.testing.assert_array_equal(session.positions, old_positions)
    invalid = observation(2)
    invalid[OBS_KEYS[-1]] = invalid[OBS_KEYS[-1]].astype(np.float32)
    with pytest.raises(ValueError, match='uint8'):
        session.act(invalid)
    np.testing.assert_array_equal(session.positions, old_positions)
    assert (session.act(obs) == 11).all() and len(predictor.calls) == 1
    session.act(obs)
    assert len(predictor.calls) == 2
    obs['task_id'] = np.array([9, 9])
    session.act(obs)
    assert len(predictor.calls) == 3 and predictor.calls[-1][0].shape[0] == 1
    assert session.act(observation(1)).shape == (1, 23)
    session.reset()
    assert session.plans is None
    del obs['task_id']
    with pytest.raises(ValueError, match='requires task_id'):
        session.act(obs)
    one = checkpoint_stub()
    one['task_map'] = {7: 'first'}
    single = Session(predictor, one, 2)
    assert single.act(obs).shape == (2, 23)
    with pytest.raises(ValueError, match='Unseen task name'):
        Session(predictor, one, 2, 'second')


def test_temporal_aggregation_upstream_weights_zeros_expiry_and_transactions():
    zero_session = Session(lambda qpos, images: np.zeros((len(qpos), 4, 23)),
                           checkpoint_stub(), temporal_agg=True)
    assert not zero_session.act(observation()).any()
    zero_session.predictor = lambda qpos, images: np.ones((len(qpos), 4, 23))
    np.testing.assert_allclose(zero_session.act(observation()), np.exp(-.01) / (1 + np.exp(-.01)))
    session = Session(CountingPredictor(), checkpoint_stub(), temporal_agg=True)
    history = []
    for step in range(9):
        value = 0 if step == 0 else step * 10
        predictions = np.broadcast_to(value + np.arange(4)[:, None], (4, 23)).copy()
        predictions[:, 0] = 0
        session.predictor = lambda qpos, images, p=predictions: np.repeat(p[None], len(qpos), axis=0)
        history.append(predictions)
        candidates = np.stack([plan[step - i] for i, plan in enumerate(history) if step - i < 4])
        weights = np.exp(-.01 * np.arange(len(candidates)))
        weights /= weights.sum()
        expected = (candidates * weights[:, None]).sum(0)
        actual = session.act(observation(value=value))[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        assert actual[0] == 0 and len(session.history) <= 3
    saved = [(p.copy(), v.copy()) for p, v in session.history]
    session.predictor = lambda qpos, images: np.full((len(qpos), 4, 23), np.nan)
    with pytest.raises(ValueError, match='Invalid model output'):
        session.act(observation())
    for (p, v), (expected_p, expected_v) in zip(session.history, saved):
        np.testing.assert_array_equal(p, expected_p)
        np.testing.assert_array_equal(v, expected_v)
    session.reset()
    assert session.history == [] and session.task_ids is None


def test_temporal_aggregation_per_slot_task_reset_batch_resize_and_connections():
    predictor = CountingPredictor()
    a = Session(predictor, checkpoint_stub(), temporal_agg=True)
    b = Session(predictor, checkpoint_stub(), temporal_agg=True)
    a.act(observation(2, np.array([7, 9]), value=10))
    b.act(observation(value=30))
    result = a.act(observation(2, np.array([9, 9]), value=20))
    assert (result[0] == 20).all()
    weights = np.exp(-.01 * np.arange(2))
    assert result[1, 0] == pytest.approx((11 * weights[0] + 20 * weights[1]) / weights.sum())
    a.reset()
    assert b.act(observation(value=40))[0, 0] == pytest.approx((31 * weights[0] + 40 * weights[1]) / weights.sum())
    assert (a.act(observation(value=50)) == 50).all()
    assert (b.act(observation(2, np.array([7, 9]), value=60)) == 60).all()


def test_msgpack_round_trip_rejects_objects():
    data = {'array': np.arange(12, dtype=np.float32).reshape(2, 6), 'scalar': np.int64(9)}
    out = unpackb(packb(data))
    np.testing.assert_array_equal(data['array'], out['array'])
    assert out['scalar'] == 9
    with pytest.raises(ValueError):
        packb(np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        unpackb(packb({b'__ndarray__': True, b'dtype': 'O', b'data': b'', b'shape': [0]}))


def test_network_handshake_health_reset_batch_and_connection_isolation():
    async def run():
        predictor = CountingPredictor()
        server = B1KServer(checkpoint_stub(), predictor, action_horizon=2)
        async with websockets.asyncio.server.serve(server.handler, '127.0.0.1', 0,
                                                   process_request=server.health) as running:
            port = running.sockets[0].getsockname()[1]
            reader, writer = await asyncio.open_connection('127.0.0.1', port)
            writer.write(b'GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
            await writer.drain()
            response = await reader.read()
            assert b'200 OK' in response and response.endswith(b'OK\n')
            writer.close()
            await writer.wait_closed()
            url = f'ws://127.0.0.1:{port}'
            async with websockets.asyncio.client.connect(url) as a, websockets.asyncio.client.connect(url) as b:
                for client in (a, b):
                    metadata = msgpack.unpackb(await client.recv(), raw=False)
                    assert metadata['action_dim'] == 23 and metadata['task_map'] == {'7': 'first', '9': 'second'}
                async def request(client, obs):
                    await client.send(packb(obs))
                    return unpackb(await client.recv())['action']
                x, y = await asyncio.gather(request(a, observation(value=10)), request(b, observation(value=20)))
                assert (x == 10).all() and (y == 20).all()
                await a.send(packb({'reset': True}))
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(a.recv(), .05)
                assert (await request(b, observation(value=20)) == 21).all()
                assert (await request(a, observation(value=30)) == 30).all()
                assert (await request(a, observation(2, np.array([7, 9]), value=40)) == 40).all()
                assert (await request(a, observation(2, np.array([9, 9]), value=50)))[0, 0] == 50
                await a.send(packb(observation(task=777)))
                with pytest.raises(websockets.exceptions.ConnectionClosedError):
                    await a.recv()
    asyncio.run(run())
