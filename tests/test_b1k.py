"""Boundary, normalization, upstream numerical, and BEHAVIOR protocol regressions."""

import asyncio
import json
import msgpack
from unittest.mock import MagicMock, patch
import sys
from types import SimpleNamespace

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
                         SplitBatchSampler, preprocess_image, preprocess_state)
from b1k_frame_cache import (FrameCacheReader, build_frame_cache, dequantize_images, entry_is_valid, quantize_image,
                             selected_videos, verify_frame_cache)
from b1k_server import B1KServer, Session, packb, unpackb
from b1k_training import (atomic_save, committed_checkpoints, configure_cpu_threads, consume_batch, load_checkpoint,
                          make_policy, match_optimizer_state_layout, optimizer_batches, parser, prepare_images,
                          prune_checkpoints, run_lock, save_eval_checkpoint, save_full_checkpoint, train, wandb_run)
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


@pytest.fixture
def fake_wandb(monkeypatch):
    sdk = MagicMock()
    sdk.login.return_value = True
    runs = []

    def initialize(**kwargs):
        run = SimpleNamespace(id=kwargs['id'], project=kwargs['project'], entity=kwargs['entity'] or 'test-entity',
                              name=kwargs['name'], settings=SimpleNamespace(mode=kwargs['mode']),
                              config=MagicMock(), define_metric=MagicMock(), log=MagicMock(), finish=MagicMock())
        runs.append(run)
        return run

    sdk.init.side_effect = initialize
    monkeypatch.setitem(sys.modules, 'wandb', sdk)
    monkeypatch.setenv('WANDB_API_KEY', 'mock-api-key')
    return sdk, runs


@pytest.mark.parametrize('policy_class', ['ACT', 'CNNMLP'])
def test_checkpoint_resume_matches_uninterrupted_and_serves_without_dataset(tiny_root, tmp_path, policy_class,
                                                                          fake_wandb):
    common = ['--dataset-path', str(tiny_root), '--batch-size', '2', '--num-workers', '0',
              '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
              '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
              '--no-pretrained-backbone', '--save-every', '1', '--dropout', '0.1', '--policy-class', policy_class,
              '--save-total-limit', '2', '--export-every', '1']
    if policy_class == 'CNNMLP':
        common += ['--image-size', '416', '416', '--batch-size', '1']
    full = tmp_path / 'full'
    split = tmp_path / 'split'
    train(parser().parse_args(common + ['--output-dir', str(full), '--max-steps', '3']))
    common += ['--wandb-mode', 'online', '--wandb-project', 'test-project', '--wandb-entity', 'test-entity',
               '--loader-batch-size', '1']
    train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '2']))
    resumed_path = train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '3',
                           '--resume', str(split / 'step_00000002.pt')]))
    resumed = load_checkpoint(resumed_path)
    expected = load_checkpoint(full / 'step_00000003.pt')
    sdk, runs = fake_wandb
    assert len(runs) == 2 and runs[0].id == runs[1].id == resumed['wandb']['id']
    assert sdk.init.call_args.kwargs['resume'] == 'must'
    assert [call.kwargs['step'] for run in runs for call in run.log.call_args_list] == [1, 2, 3]
    assert all(run.finish.call_count == 1 for run in runs)
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)
    assert resumed['step'] == 3 and resumed['optimizer']['state']
    assert [p.name for p in committed_checkpoints(split)] == ['step_00000002.pt', 'step_00000003.pt']
    assert len(committed_checkpoints(split / 'export_queue/eval')) == 3
    assert [p.name for p in committed_checkpoints(split / 'export_queue/full')] == ['step_00000003.pt']
    assert (split / 'export_queue/full/step_00000003.pt').stat().st_ino == resumed_path.stat().st_ino
    exported_path = split / 'export_queue/eval/step_00000003.pt'
    exported = load_checkpoint(exported_path)
    assert exported['checkpoint_type'] == 'eval'
    assert not {'optimizer', 'torch_rng', 'cuda_rng'} & exported.keys()
    for key in ('model_config', 'adapter_config', 'normalization', 'task_map', 'step'):
        assert exported[key] == resumed[key]
    for key, tensor in exported['model'].items():
        assert tensor.device.type == 'cpu'
        torch.testing.assert_close(tensor, resumed['model'][key], atol=0, rtol=0)
    with pytest.raises(ValueError, match='eval-only.*full'):
        train(parser().parse_args(common + ['--output-dir', str(tmp_path / 'bad-resume'), '--max-steps', '4',
                                           '--resume', str(exported_path)]))
    records = [json.loads(line) for line in (split / 'metrics.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == [1, 2, 3]
    assert all(record['timing/data_wait_s'] >= 0 and record['data/parquet_s'] > 0
               and record['data/video_decode_s'] > 0 and record['gpu/allocated_bytes'] == 0 for record in records)
    with pytest.raises(FileExistsError):
        train(parser().parse_args(common + ['--output-dir', str(full), '--max-steps', '3']))
    with pytest.raises(FileExistsError, match='existing snapshots'):
        train(parser().parse_args(common + ['--output-dir', str(full), '--max-steps', '4',
                                           '--resume', str(full / 'step_00000002.pt')]))
    tiny_root.rename(tmp_path / 'dataset_hidden')
    from b1k_server import ACTPredictor
    predictor = ACTPredictor(exported, 'cpu')
    session = Session(predictor, exported, 2 if policy_class == 'ACT' else 1)
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


def test_atomic_checkpoints_retention_and_failed_save(tmp_path, monkeypatch):
    output = tmp_path / 'checkpoints'
    output.mkdir()
    stale = output / 'step_00000000.tmp'
    stale.write_bytes(b'incomplete')
    (output / 'step_unknown.pt').write_bytes(b'unrelated')
    eval_queue = output / 'export_queue/eval'
    eval_queue.mkdir(parents=True)
    atomic_save({'step': 1}, eval_queue / 'step_00000001.pt')
    for step in range(1, 6):
        latest = save_full_checkpoint({'step': step}, output, save_total_limit=3, export_queue=True)
    assert [p.name for p in committed_checkpoints(output)] == [f'step_{step:08d}.pt' for step in (3, 4, 5)]
    assert (output / 'latest.pt').resolve() == latest
    assert (output / 'export_queue/full' / latest.name).stat().st_ino == latest.stat().st_ino
    assert len(committed_checkpoints(output / 'export_queue/full')) == 1
    assert stale.exists() and (output / 'step_unknown.pt').exists()
    assert (eval_queue / 'step_00000001.pt').exists()
    with pytest.raises(FileExistsError):
        save_full_checkpoint({'step': 5}, output, save_total_limit=3, export_queue=True)
    before = [p.name for p in committed_checkpoints(output)]
    real_save = torch.save

    def fail_save(value, stream):
        stream.write(b'partial')
        raise OSError('simulated interrupted save')

    monkeypatch.setattr(torch, 'save', fail_save)
    with pytest.raises(OSError, match='interrupted'):
        save_full_checkpoint({'step': 6}, output, save_total_limit=1, export_queue=True)
    assert [p.name for p in committed_checkpoints(output)] == before
    assert not (output / 'step_00000006.pt').exists()
    assert not list(output.glob('.*.tmp'))
    assert (output / 'latest.pt').resolve() == latest
    monkeypatch.setattr(torch, 'save', real_save)
    prune_checkpoints(output, 0)
    assert [p.name for p in committed_checkpoints(output)] == before


def test_run_lock_is_exclusive_and_released_after_failure(tmp_path):
    with pytest.raises(ValueError, match='test'):
        with run_lock(tmp_path):
            with pytest.raises(RuntimeError, match='Another trainer holds'):
                with run_lock(tmp_path):
                    pytest.fail('A second trainer acquired the lock')
            raise ValueError('test')
    with run_lock(tmp_path):
        assert (tmp_path / 'run.lock').exists()


def test_wandb_modes_fail_closed_and_preserve_identity(tmp_path, fake_wandb, monkeypatch):
    sdk, runs = fake_wandb
    args = parser().parse_args(['--dataset-path', '/unused', '--output-dir', str(tmp_path)])
    with wandb_run(args, tmp_path, None) as (run, identity):
        assert run is identity is None
    sdk.init.assert_not_called()
    args.wandb_mode = 'online'
    monkeypatch.delenv('WANDB_API_KEY')
    with pytest.raises(RuntimeError, match='WANDB_API_KEY'):
        with wandb_run(args, tmp_path, None):
            pass
    monkeypatch.setenv('WANDB_API_KEY', 'mock-api-key')
    sdk.login.return_value = False
    with pytest.raises(RuntimeError, match='authentication failed'):
        with wandb_run(args, tmp_path, None):
            pass
    sdk.init.assert_not_called()
    sdk.login.return_value = True
    with wandb_run(args, tmp_path, None) as (_, saved):
        assert saved['mode'] == 'online'
    sdk.login.assert_called_with(verify=True)
    assert json.loads((tmp_path / 'wandb.json').read_text()) == saved
    copied = tmp_path / 'resumed'
    with wandb_run(args, copied, {'wandb': saved}) as (_, resumed):
        assert resumed == saved
    assert sdk.init.call_args.kwargs['resume'] == 'must'
    args.wandb_id = 'different'
    with pytest.raises(ValueError, match='preserve W&B identity'):
        with wandb_run(args, copied, {'wandb': saved}):
            pass
    args.wandb_id = None
    args.wandb_mode = 'offline'
    sdk.login.reset_mock()
    with wandb_run(args, copied, {'wandb': saved}) as (_, offline):
        assert offline['id'] == saved['id']
    sdk.login.assert_not_called()
    args.wandb_mode = 'online'
    sdk.init.side_effect = None
    sdk.init.return_value = runs[-1]
    with pytest.raises(RuntimeError, match='non-online'):
        with wandb_run(args, copied, {'wandb': offline}):
            pass
    runs[-1].finish.assert_called_with(exit_code=1)


def test_data_timing_worker_caps_and_prefetch(tiny_root, monkeypatch):
    from functools import partial
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), profile_reads=True)
    dataset.stats = dataset.compute_stats()
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=1, prefetch_factor=2,
                multiprocessing_context='spawn', worker_init_fn=partial(configure_cpu_threads,
                torch_threads=1, arrow_threads=1, opencv_threads=1))
    batches = list(loader)
    assert len(batches) == 5
    assert all(batch[4]['data/sample_s'].min() > 0 for batch in batches)
    cv2 = MagicMock()
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    configure_cpu_threads(torch_threads=1, arrow_threads=1, opencv_threads=1)
    assert torch.get_num_threads() == pa.cpu_count() == pa.io_thread_count() == 1
    cv2.setNumThreads.assert_called_once_with(1)
    dataset.close()


def test_split_loader_reassembles_identical_order_and_resume(tiny_root):
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), profile_reads=True)
    dataset.stats = dataset.compute_stats()
    whole_sampler = StepBatchSampler(dataset, 5, 0, 3, 17)
    split_sampler = SplitBatchSampler(whole_sampler, 2)
    assert len(split_sampler) == 9
    assert [index for batch in whole_sampler for index in batch] == [index for batch in split_sampler for index in batch]
    reference = list(torch.utils.data.DataLoader(dataset, batch_sampler=whole_sampler))
    sliced = torch.utils.data.DataLoader(dataset, batch_sampler=split_sampler)
    actual = list(optimizer_batches(sliced, 5, 2))
    resumed_sampler = SplitBatchSampler(StepBatchSampler(dataset, 5, 1, 3, 17), 2)
    resumed = list(optimizer_batches(torch.utils.data.DataLoader(dataset, batch_sampler=resumed_sampler), 5, 2))
    for expected, batch in zip(reference, actual):
        for tensor, other in zip(expected[:4], batch[:4]):
            torch.testing.assert_close(tensor, other, atol=0, rtol=0)
    for expected, batch in zip(actual[1:], resumed):
        for tensor, other in zip(expected[:4], batch[:4]):
            torch.testing.assert_close(tensor, other, atol=0, rtol=0)
    dataset.close()


def test_frame_cache_matches_native_decoding_and_detects_stale_entries(tiny_root, tmp_path):
    from functools import partial
    cache = tmp_path / 'cache'
    native = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16))
    with pytest.raises(ValueError, match='read-only dataset'):
        build_frame_cache(native, tiny_root / 'cache', log=lambda *_: None)
    videos = build_frame_cache(native, cache, workers=2, log=lambda *_: None)
    assert videos == selected_videos(native) and len(videos) == 3
    assert verify_frame_cache(native, cache, samples=10, log=lambda *_: None) == 30
    manifest = json.loads(next(cache.rglob('*.json')).read_text())
    assert manifest['frames'] == 20 and manifest['image_size'] == [16, 16]
    assert build_frame_cache(native, cache, log=lambda *_: None) == videos  # valid entries are reused
    native.stats = native.compute_stats()
    cached = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), frame_cache=cache, profile_reads=True)
    cached.stats = native.stats
    assert cached._table['action'].shape == (10, 23) and cached._frame_rows.shape == (10, 3)
    for index in range(len(native)):
        expected, actual = native[index], cached[index]
        assert actual[0].dtype == torch.uint8 and actual[0].shape == (3, 16, 16, 3)
        images = dequantize_images(actual[0])
        assert images.shape == expected[0].shape and images.stride()[-3] == 1  # channels-last per camera
        assert (images - expected[0]).abs().max() <= 0.5 / 255 + 1e-6
        assert torch.equal(actual[0], torch.stack([torch.from_numpy(quantize_image(image)) for image in expected[0]]))
        for column in (1, 2, 3):
            torch.testing.assert_close(expected[column], actual[column], atol=0, rtol=0)
        assert actual[4]['data/video_decode_s'] == 0 and actual[4]['data/frame_cache_s'] > 0
    # Camera offsets: cached frames are the same frames the native reader selects (raw pixels differ per camera).
    raw, _, _, _, _ = native.raw_sample(99, 1)
    assert [int(x[0, 0, 0]) for x in raw] == [100, 121, 142]
    assert [int(cached.sample_at(99, 1)[0][c, 0, 0, 0]) for c in range(3)] == [100, 121, 142]
    loader = torch.utils.data.DataLoader(cached, batch_size=4, num_workers=1, multiprocessing_context='spawn',
                                         worker_init_fn=partial(configure_cpu_threads, torch_threads=1))
    batches = list(loader)
    assert sum(len(batch[0]) for batch in batches) == 10 and batches[0][0].dtype == torch.uint8
    torch.testing.assert_close(torch.cat([batch[2] for batch in batches]), torch.stack([native[i][2] for i in range(10)]))
    cached.close()
    with pytest.raises(FileNotFoundError, match='No frame cache entry'):
        B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), frame_cache=tmp_path / 'missing')
    with pytest.raises(ValueError, match='Stale or mismatched'):
        B1KDataset(tiny_root, chunk_size=4, image_size=(8, 8), frame_cache=cache)
    video = videos[0]
    import os
    os.utime(video, ns=(video.stat().st_atime_ns, video.stat().st_mtime_ns + 1))
    assert not entry_is_valid(cache / video.relative_to(tiny_root).with_suffix(''), video, (16, 16))
    with pytest.raises(ValueError, match='Stale or mismatched'):
        FrameCacheReader(cache, tiny_root, (16, 16)).open(video)
    reader = FrameCacheReader(cache, tiny_root, (16, 16))
    other = videos[1]
    _, pts = reader.open(other)
    np.testing.assert_array_equal(reader.positions(other, pts[[3, 0, 19]] + 0.004), [3, 0, 19])
    with pytest.raises(ValueError, match='timestamp mismatch'):
        reader.positions(other, [pts[3] + 0.05])
    native.close()


def test_frame_cache_training_matches_uninterrupted_and_records_provenance(tiny_root, tmp_path):
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(32, 32))
    build_frame_cache(dataset, tmp_path / 'cache', log=lambda *_: None)
    dataset.close()
    common = ['--dataset-path', str(tiny_root), '--batch-size', '2', '--num-workers', '0', '--device', 'cpu',
              '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32', '--dim-feedforward', '64',
              '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4', '--no-pretrained-backbone', '--save-every', '1',
              '--frame-cache', str(tmp_path / 'cache')]
    full = tmp_path / 'full'
    split = tmp_path / 'split'
    train(parser().parse_args(common + ['--output-dir', str(full), '--max-steps', '3']))
    common += ['--loader-batch-size', '1']
    train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '2']))
    resumed_path = train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '3',
                                                       '--resume', str(split / 'step_00000002.pt')]))
    resumed, expected = load_checkpoint(resumed_path), load_checkpoint(full / 'step_00000003.pt')
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)
    run = json.loads((full / 'run.json').read_text())
    assert run['train_config']['frame_cache'] == str(tmp_path / 'cache')
    assert run['adapter_config']['image_resize'] == 'bilinear_antialias'
    records = [json.loads(line) for line in (full / 'metrics.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == [1, 2, 3]
    assert all(record['data/video_decode_s'] == 0 and record['data/frame_cache_s'] > 0 and record['data/parquet_s'] > 0
               and record['timing/data_wait_s'] >= 0 for record in records)
    # The same steps without the cache differ only by the uint8 rounding of the frames.
    plain = tmp_path / 'plain'
    train(parser().parse_args([arg for arg in common if arg not in ('--frame-cache', str(tmp_path / 'cache'))]
                              + ['--output-dir', str(plain), '--max-steps', '1', '--loader-batch-size', '2']))
    first = [json.loads(line) for line in (plain / 'metrics.jsonl').read_text().splitlines()][0]
    assert first['loss'] == pytest.approx(records[0]['loss'], rel=1e-2)


def test_fused_attention_matches_explicit_attention_weights_path():
    from detr.models.transformer import use_fused_attention
    torch.set_num_threads(1)
    torch.manual_seed(3)
    policy = make_policy(small_model_config(), 'cpu')
    layers = [module for module in policy.modules() if hasattr(module, 'attention')]
    assert len(layers) == 3 and all(layer.attention == 'auto' for layer in layers)
    assert not use_fused_attention('auto')  # fp32 without autocast keeps upstream's explicit attention
    with torch.autocast('cpu', dtype=torch.bfloat16):
        assert use_fused_attention('auto')
    with pytest.raises(ValueError, match='attention mode'):
        make_policy(small_model_config(), 'cpu', attention='other')
    qpos, images = torch.randn(2, 27), torch.rand(2, 3, 3, 32, 32)
    actions = torch.randn(2, 4, 23)
    pad = torch.tensor([[False, False, True, True], [False, False, False, False]])
    original = torch.nn.MultiheadAttention.forward
    calls = []

    def recording(self, *args, **kwargs):
        calls.append(kwargs.get('need_weights'))
        return original(self, *args, **kwargs)

    def run(mode):
        for layer in layers:
            layer.attention = mode
        calls.clear()
        with patch.object(torch.nn.MultiheadAttention, 'forward', recording):
            policy.zero_grad(set_to_none=True)
            torch.manual_seed(5)
            losses = policy(qpos, images, actions, pad)
            losses['loss'].backward()
        assert calls and all(call is (mode == 'explicit') for call in calls)
        return ({k: v.detach().clone() for k, v in losses.items()},
                [p.grad.clone() for p in policy.parameters() if p.grad is not None])
    reference_losses, reference_grads = run('explicit')
    losses, grads = run('fused')
    for key in losses:
        torch.testing.assert_close(losses[key], reference_losses[key], atol=1e-6, rtol=1e-6)
    assert len(grads) == len(reference_grads)
    for grad, reference in zip(grads, reference_grads):
        torch.testing.assert_close(grad, reference, atol=1e-6, rtol=1e-5)
    policy.eval()
    with torch.no_grad():
        for layer in layers:
            layer.attention = 'explicit'
        expected = policy(qpos, images)
        for layer in layers:
            layer.attention = 'fused'
        torch.testing.assert_close(policy(qpos, images), expected, atol=1e-6, rtol=1e-6)


def test_skipping_unused_decoder_layers_is_exact():
    from b1k_training import unused_gradients
    torch.set_num_threads(1)
    config = dict(small_model_config(), dec_layers=3)
    qpos, images = torch.randn(2, 27), torch.rand(2, 3, 3, 32, 32)
    actions = torch.randn(2, 4, 23)
    pad = torch.tensor([[False, False, True, True], [False, False, False, False]])
    policies = {}
    for skip in (False, True):
        torch.manual_seed(11)
        policies[skip] = make_policy(config, 'cpu', skip_unused_decoder_layers=skip)
    upstream, skipped = policies[False], policies[True]
    assert upstream.model.decoder_layers_used is None and skipped.model.decoder_layers_used == 1
    assert list(upstream.state_dict()) == list(skipped.state_dict())
    for key, value in upstream.state_dict().items():
        torch.testing.assert_close(value, skipped.state_dict()[key], atol=0, rtol=0)
    unused = skipped.model.unused_parameters()
    assert len(unused) == len(list(skipped.model.transformer.decoder.layers[1:].parameters())) > 0
    assert not upstream.model.unused_parameters()
    # Predictions are identical; upstream autograd gives the skipped layers exactly-zero gradients.
    for policy in policies.values():
        policy.eval()
    with torch.no_grad():
        torch.testing.assert_close(upstream(qpos, images), skipped(qpos, images), atol=0, rtol=0)
    for policy in policies.values():
        policy.train()
    reference = {}
    for skip, policy in policies.items():
        torch.manual_seed(4)
        losses = policy(qpos, images, actions, pad)
        losses['loss'].backward()
        reference[skip] = losses
    torch.testing.assert_close(reference[False]['loss'], reference[True]['loss'], atol=0, rtol=0)
    unused_names = {name for name, p in skipped.model.named_parameters() if any(p is q for q in unused)}
    assert all(name.startswith('transformer.decoder.layers.') and not name.startswith('transformer.decoder.layers.0.')
               for name in unused_names)
    for (name, expected), (_, actual) in zip(upstream.model.named_parameters(), skipped.model.named_parameters()):
        if name in unused_names:
            assert expected.grad is not None and not expected.grad.any() and actual.grad is None
        elif expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(expected.grad, actual.grad, atol=0, rtol=0)
    # Two optimizer steps with the training loop's zero-gradient substitution reproduce every parameter,
    # including the weight-decayed unused layers, and the optimizer state.
    optimizers = {skip: policy.configure_optimizers() for skip, policy in policies.items()}
    substitutes = unused_gradients(skipped)
    assert len(substitutes) == len(unused) and not unused_gradients(upstream)
    for _ in range(2):
        for skip, policy in policies.items():
            optimizers[skip].zero_grad(set_to_none=True)
            torch.manual_seed(6)
            policy(qpos, images, actions, pad)['loss'].backward()
            if skip:
                for parameter, zeros in substitutes:
                    parameter.grad = zeros
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'), error_if_nonfinite=True)
            optimizers[skip].step()
    for key, value in upstream.state_dict().items():
        torch.testing.assert_close(value, skipped.state_dict()[key], atol=0, rtol=0)
    upstream_state, skipped_state = optimizers[False].state_dict()['state'], optimizers[True].state_dict()['state']
    assert upstream_state.keys() == skipped_state.keys()
    for index in upstream_state:
        for key, value in upstream_state[index].items():
            torch.testing.assert_close(value, skipped_state[index][key], atol=0, rtol=0)


def test_image_layouts_optimizer_state_layout_and_batch_assembly(tiny_root):
    frames = torch.randint(0, 256, (2, 3, 8, 8, 3), dtype=torch.uint8)
    images = prepare_images(frames)
    assert images.shape == (2, 3, 3, 8, 8) and images.dtype == torch.float32
    torch.testing.assert_close(images, frames.permute(0, 1, 4, 2, 3).float() / 255)
    assert images[:, 0].is_contiguous(memory_format=torch.channels_last) and not images.is_contiguous()
    assert prepare_images(frames, channels_last=False).is_contiguous()
    floats = torch.rand(2, 3, 3, 8, 8)
    relaid = prepare_images(floats)
    torch.testing.assert_close(relaid, floats)
    assert relaid[:, 1].is_contiguous(memory_format=torch.channels_last)
    assert prepare_images(floats, channels_last=False) is floats
    with pytest.raises(ValueError, match='uint8'):
        dequantize_images(floats)
    conv = torch.nn.Conv2d(3, 4, 3)
    optimizer = torch.optim.AdamW(conv.parameters(), lr=1e-3)
    conv(torch.rand(1, 3, 8, 8)).sum().backward()
    optimizer.step()
    saved = {k: v.clone() for k, v in optimizer.state[conv.weight].items()}
    conv.to(memory_format=torch.channels_last)
    match_optimizer_state_layout(optimizer)
    for key, value in optimizer.state[conv.weight].items():
        torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
        if value.dim() == 4:
            assert value.stride() == conv.weight.stride() and value.is_contiguous(memory_format=torch.channels_last)
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), profile_reads=True)
    dataset.stats = dataset.compute_stats()
    sampler = SplitBatchSampler(StepBatchSampler(dataset, 5, 0, 2, 17), 2)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    device = torch.device('cpu')
    batches = list(optimizer_batches(loader, 5, 2, device=device))
    reference = list(optimizer_batches(torch.utils.data.DataLoader(dataset, batch_sampler=sampler), 5, 2))
    assert len(batches) == 2
    for batch, expected in zip(batches, reference):
        consumed = consume_batch(batch, None, channels_last=True)
        for tensor, other in zip(consumed[1:4], expected[1:4]):
            torch.testing.assert_close(tensor, other, atol=0, rtol=0)
        torch.testing.assert_close(consumed[0], expected[0], atol=0, rtol=0)
        assert set(consumed[4]) == set(expected[4]) and len(consumed[4]['data/sample_s']) == 5
    dataset.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')
def test_device_batch_assembly_on_side_stream_matches_host(tiny_root, tmp_path):
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16))
    build_frame_cache(dataset, tmp_path / 'cache', log=lambda *_: None)
    dataset.close()
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), frame_cache=tmp_path / 'cache', profile_reads=True)
    dataset.stats = dataset.compute_stats()
    device = torch.device('cuda')
    stream = torch.cuda.Stream(device)
    for loader_batch in (2, None):
        sampler = StepBatchSampler(dataset, 5, 0, 3, 17)
        split = SplitBatchSampler(sampler, loader_batch) if loader_batch else sampler
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=split, pin_memory=True)
        host = list(optimizer_batches(torch.utils.data.DataLoader(dataset, batch_sampler=split), 5, loader_batch))
        batches = optimizer_batches(loader, 5, loader_batch, device=device, stream=stream)
        upcoming = next(batches)
        for expected in host:
            batch = upcoming
            upcoming = next(batches, None)  # prefetched while the previous batch is consumed
            images, qpos, actions, is_pad, timings = consume_batch(batch, stream, channels_last=True)
            assert images.device.type == 'cuda' and images[:, 0].is_contiguous(memory_format=torch.channels_last)
            # uint8 / 255 may round differently by one ulp between the CUDA and CPU kernels
            torch.testing.assert_close(images.cpu(), prepare_images(expected[0]), atol=1e-7, rtol=0)
            for tensor, other in zip((qpos, actions, is_pad), expected[1:4]):
                torch.testing.assert_close(tensor.cpu(), other, atol=0, rtol=0)
            assert timings['data/sample_s'].shape == expected[4]['data/sample_s'].shape and timings['data/sample_s'].min() > 0
        assert upcoming is None
    dataset.close()


def test_save_first_step_and_exact_statistics_guard(tiny_root, tmp_path, monkeypatch):
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters())

        def forward(self, qpos, images, actions, is_pad):
            return {'loss': self.weight.square()}

    monkeypatch.setattr('b1k_training.make_policy', lambda *args, **kwargs: TinyPolicy())
    common = ['--dataset-path', str(tiny_root), '--device', 'cpu', '--num-workers', '0',
              '--batch-size', '2', '--image-size', '16', '16', '--save-first-step', '--save-every', '10',
              '--export-every', '2', '--save-total-limit', '3', '--max-steps', '3']
    output = tmp_path / 'first-step'
    train(parser().parse_args(common + ['--output-dir', str(output)]))
    assert [p.name for p in committed_checkpoints(output)] == ['step_00000001.pt', 'step_00000003.pt']
    assert [p.name for p in committed_checkpoints(output / 'export_queue/eval')] == ['step_00000002.pt']
    exported = load_checkpoint(output / 'export_queue/eval/step_00000002.pt')
    metadata = {key: exported[key] for key in ('model_config', 'adapter_config', 'task_map', 'normalization')}
    policy = TinyPolicy()
    policy.load_state_dict(exported['model'])
    save_eval_checkpoint(metadata, policy, 2, output)
    with torch.no_grad():
        policy.weight.add_(1)
    with pytest.raises(FileExistsError, match='eval export differs'):
        save_eval_checkpoint(metadata, policy, 2, output)
    smoke = tmp_path / 'smoke'
    path = train(parser().parse_args(common + ['--output-dir', str(smoke), '--stats-max-frames', '2']))
    assert load_checkpoint(path)['normalization']['approximate']
    with pytest.raises(ValueError, match='Full-task statistics required'):
        train(parser().parse_args(common + ['--output-dir', str(smoke), '--resume', str(path), '--max-steps', '4']))


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
