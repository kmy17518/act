"""Wrist-goal control (--goal-mismatch-views): the chosen goal views show another episode of the same task, drawn per
sample from a separate stream; frames, batches, initialization and every other goal view are those of the run
without the option."""

import json
from pathlib import Path
import pickle
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from b1k_dataset import VIDEO_KEYS, B1KDataset, StepBatchSampler
from b1k_training import load_checkpoint, parser, train

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/b1k'))
import isg_tier_a  # noqa: E402

VIEWS = ['zed_link', 'left_realsense_link', 'right_realsense_link']
WRISTS = VIEWS[1:]
LENGTH = 3
EPISODE_TASK = [0, 0, 0, 0, 1, 1, 1, 1]  # episodes 0-3: reach, 4-7: push
SPLIT = {'reach': {'train': [0, 1], 'held_out': [2, 3]}, 'push': {'train': [4, 5], 'held_out': [6, 7]}}


@pytest.fixture
def mismatch_root(tmp_path):
    """Eight 3-frame episodes of two tasks. Every camera frame is constant at (global frame * 10 + camera), so a goal
    image names its episode and camera (`owner`). The split trains on 0, 1, 4, 5 and holds out 2, 3, 6, 7."""
    root = tmp_path / 'mismatch'
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    (root / 'isg_meta').mkdir()
    info = {'codebase_version': 'v3.0', 'fps': 10,
            'features': {'action': {'shape': [23]}, 'observation.state': {'shape': [61]}},
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mkv'}
    (root / 'meta/info.json').write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist([{'task_index': 0, 'task': 'reach'}, {'task_index': 1, 'task': 'push'}]),
                   root / 'meta/tasks.parquet')
    for c, key in enumerate(VIDEO_KEYS):
        video = root / info['video_path'].format(video_key=key, chunk_index=0, file_index=0)
        video.parent.mkdir(parents=True)
        with av.open(str(video), 'w') as container:
            stream = container.add_stream('ffv1', rate=10)
            stream.width = stream.height = 16
            stream.pix_fmt = 'bgr0'
            for i in range(len(EPISODE_TASK) * LENGTH):
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.full((16, 16, 3), i * 10 + c, np.uint8),
                                                                       format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    rows, episodes = [], []
    for e, task in enumerate(EPISODE_TASK):
        ep = {'episode_index': e, 'task_index': task, 'length': LENGTH, 'dataset_from_index': e * LENGTH,
              'dataset_to_index': (e + 1) * LENGTH, 'data/chunk_index': 0, 'data/file_index': 0}
        for key in VIDEO_KEYS:
            ep.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': 0,
                       f'videos/{key}/from_timestamp': e * LENGTH / 10, f'videos/{key}/to_timestamp': (e + 1) * LENGTH / 10})
        episodes.append(ep)
        for frame in range(LENGTH):
            state = np.arange(61, dtype=np.float32) * .01 + frame + e
            action = np.sin(np.arange(23, dtype=np.float32) + frame + e)
            rows.append({'episode_index': e, 'task_index': task, 'index': e * LENGTH + frame, 'frame_index': frame,
                         'timestamp': frame / 10, 'observation.state': state.tolist(), 'action': action.tolist()})
    pq.write_table(pa.Table.from_pylist(rows), root / 'data/chunk-000/file-000.parquet', row_group_size=LENGTH)
    pq.write_table(pa.Table.from_pylist(episodes), root / 'meta/episodes/chunk-000/file-000.parquet')
    (root / 'isg_meta/train_split.json').write_text(json.dumps({'format': 'isg-episode-split/v1', 'name': 'mismatch',
                                                                'tasks': SPLIT}))
    return root


def owner(image):
    """(episode, camera) whose last frame a goal image is."""
    value = int(image[0, 0, 0])
    return value // 10 // LENGTH, value % 10


def dataset(root, **kwargs):
    data = B1KDataset(root, chunk_size=2, image_size=(16, 16), goal_views=VIEWS, task_onehot=False, **kwargs)
    data.stats = data.compute_stats()
    return data


def common(root, output, device='cpu'):
    return ['--dataset-path', str(root), '--batch-size', '2', '--num-workers', '0', '--torch-threads', '1',
            '--device', device, '--chunk-size', '2', '--image-size', '32', '32', '--hidden-dim', '32',
            '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
            '--no-pretrained-backbone', '--save-every', '1', '--export-every', '1', '--lr', '1e-3', '--lr-backbone', '1e-3',
            '--regime', 'image', '--goal-fusion', 'early', '--goal-views', *VIEWS, '--output-dir', str(output)]


def test_partner_is_another_episode_of_the_same_task_and_the_head_goal_is_unchanged(mismatch_root):
    data = dataset(mismatch_root, goal_mismatch_views=WRISTS, goal_mismatch_seed=3)
    for position, ep in enumerate(data.episodes):
        own, task = int(ep['episode_index']), int(ep['task_index'])
        seen = set()
        for step in range(30):
            for slot in range(4):
                goal = data.goal_for(position, (step, slot)).numpy()
                assert owner(goal[0]) == (own, 0)
                (left, left_camera), (right, right_camera) = owner(goal[1]), owner(goal[2])
                assert (left_camera, right_camera) == (1, 2) and left == right != own and EPISODE_TASK[left] == task
                seen.add(left)
        assert seen == {e for e, t in enumerate(EPISODE_TASK) if t == task and e != own}
        assert torch.equal(data.goal_for(position), torch.from_numpy(data.goal_table[position]))
    data.close()


def test_draws_are_deterministic_and_frame_sampling_is_identical(mismatch_root):
    data = dataset(mismatch_root, goal_mismatch_views=WRISTS, goal_mismatch_seed=3)
    again = dataset(mismatch_root, goal_mismatch_views=WRISTS, goal_mismatch_seed=3)
    worker = pickle.loads(pickle.dumps(data))
    reseeded = dataset(mismatch_root, goal_mismatch_views=WRISTS, goal_mismatch_seed=4)
    keys = [(position, step, slot) for position in range(8) for step in range(10) for slot in range(3)]
    draws = [data.mismatch_partner(*key) for key in keys]
    assert draws == [again.mismatch_partner(*key) for key in keys] == [worker.mismatch_partner(*key) for key in keys]
    assert draws != [reseeded.mismatch_partner(*key) for key in keys]
    tagged, plain = list(StepBatchSampler(data, 6, 0, 5, 7, tag=True)), list(StepBatchSampler(data, 6, 0, 5, 7))
    assert [[index for index, _, _ in batch] for batch in tagged] == plain
    assert all((step, slot) == (s, k) for s, batch in enumerate(tagged) for k, (_, step, slot) in enumerate(batch))
    index, step, slot = tagged[2][4]
    position = int(np.searchsorted(data.ends, index, side='right'))
    assert torch.equal(data[(index, step, slot)][4], data.goal_for(position, (step, slot)))
    with pytest.raises(ValueError, match='StepBatchSampler'):
        data[index]
    for item in (data, again, worker, reseeded):
        item.close()


def test_loader_batches_differ_only_in_the_mismatched_goal_views(mismatch_root):
    plain, control = dataset(mismatch_root), dataset(mismatch_root, goal_mismatch_views=WRISTS, goal_mismatch_seed=7)
    batches = [next(iter(torch.utils.data.DataLoader(
        item, batch_sampler=StepBatchSampler(item, 8, 0, 1, 7, tag=bool(item.goal_mismatch_slots))))) for item in (plain, control)]
    for a, b in zip(batches[0][:4], batches[1][:4]):  # images, qpos, actions, is_pad
        assert torch.equal(a, b)
    assert torch.equal(batches[0][4][:, 0], batches[1][4][:, 0]) and torch.equal(batches[0][5], batches[1][5])
    assert (batches[0][4][:, 1:] != batches[1][4][:, 1:]).flatten(1).any(dim=1).all()
    plain.close()
    control.close()


def test_without_the_option_goals_sampling_and_checkpoints_are_unchanged(mismatch_root, tmp_path):
    data = dataset(mismatch_root)
    assert data.goal_mismatch_slots == [] and data.task_positions == {}
    for position in range(len(data.episodes)):
        assert torch.equal(data.goal_for(position, (3, 1)), torch.from_numpy(data.goal_table[position]))
    expected = []
    for step in range(4):  # the sampler's formula before the option existed
        rng = np.random.default_rng(np.random.SeedSequence([5, step]))
        episodes = rng.integers(len(data.episodes), size=6)
        frames = [int(rng.integers(data.lengths[e])) for e in episodes]
        expected.append([int(data.starts[e]) + frame for e, frame in zip(episodes, frames)])
    assert list(StepBatchSampler(data, 6, 0, 4, 5)) == expected
    index = expected[1][2]
    assert torch.equal(data[index][4], data.goal_for(int(np.searchsorted(data.ends, index, side='right'))))
    data.close()
    checkpoint = load_checkpoint(train(parser().parse_args(common(mismatch_root, tmp_path / 'plain') + ['--max-steps', '1'])))
    assert 'goal_mismatch_views' not in checkpoint['adapter_config'] and 'goal_mismatch_views' not in checkpoint['train_config']
    assert 'mismatch_views' not in checkpoint['conditioning']['goal']


def tier_a_goals(checkpoint_path, monkeypatch):
    """Run Tier A on CPU and return (result, [(own goals, other goals) per evaluated batch]) as uint8 tensors."""
    captured = []
    real = isg_tier_a.prepare_goals
    monkeypatch.setattr(isg_tier_a, 'prepare_goals', lambda goal, *a, **k: captured.append(goal.clone()) or real(goal, *a, **k))
    result = isg_tier_a.run(isg_tier_a.parser().parse_args([str(checkpoint_path), '--frames-per-task', '4',
                                                            '--device', 'cpu', '--batch-size', '8', '--threads', '1']))
    monkeypatch.setattr(isg_tier_a, 'prepare_goals', real)
    return result, list(zip(captured[0::2], captured[1::2]))


def test_option_is_saved_resumed_and_applied_by_tier_a(mismatch_root, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    output = tmp_path / 'control'
    args = common(mismatch_root, output) + ['--goal-mismatch-views', *WRISTS]
    first_path = train(parser().parse_args(args + ['--max-steps', '1']))
    first = load_checkpoint(first_path)
    assert first['adapter_config']['goal_mismatch_views'] == WRISTS and first['train_config']['goal_mismatch_views'] == WRISTS
    assert first['conditioning']['goal']['mismatch_views'] == WRISTS
    resume = common(mismatch_root, output) + ['--resume', str(first_path), '--max-steps', '2']
    with pytest.raises(ValueError, match='--goal-mismatch-views differs from checkpoint'):
        train(parser().parse_args(resume + ['--goal-mismatch-views', WRISTS[0]]))
    resumed_path = train(parser().parse_args(resume))
    assert load_checkpoint(resumed_path)['adapter_config']['goal_mismatch_views'] == WRISTS
    for name, extra, message in (('all', ['--goal-mismatch-views', *VIEWS], 'at least one goal view'),
                                 ('subset', ['--goal-views', 'zed_link', '--goal-mismatch-views', WRISTS[0]], 'distinct views'),
                                 ('source', ['--goal-mismatch-views', *WRISTS, '--goal-source', 'goal_key'], 'episode_last')):
        with pytest.raises(ValueError, match=message):
            train(parser().parse_args(common(mismatch_root, tmp_path / name) + extra + ['--max-steps', '1']))

    result, batches = tier_a_goals(resumed_path, monkeypatch)
    assert result['goal']['goal_mismatch_views'] == WRISTS and len(batches) == 4  # 2 subsets x 2 tasks
    subsets = [SPLIT[task]['held_out'] for task in SPLIT] + [SPLIT[task]['train'] for task in SPLIT]
    for (own, other), episodes in zip(batches, subsets):
        for i in range(len(own)):
            head, (left, _), (right, _) = owner(own[i, 0].numpy()), owner(own[i, 1].numpy()), owner(own[i, 2].numpy())
            assert head[1] == 0 and head[0] in episodes and left == right != head[0] and left in episodes
            other_head = owner(other[i, 0].numpy())
            assert other_head[0] != head[0] and other_head[0] in episodes
            assert torch.equal(other[i, 1:], own[i, 1:])  # the same partner's wrist goals in both conditions

    plain_path = train(parser().parse_args(common(mismatch_root, tmp_path / 'plain') + ['--max-steps', '1']))
    result, batches = tier_a_goals(plain_path, monkeypatch)
    assert 'goal_mismatch_views' not in result['goal']
    for own, _ in batches:
        assert all(len({owner(own[i, view].numpy())[0] for view in range(3)}) == 1 for i in range(len(own)))


def test_tier_a_evaluation_only_mismatch(mismatch_root, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    plain_path = train(parser().parse_args(common(mismatch_root, tmp_path / 'plain') + ['--max-steps', '1']))
    captured = []
    real = isg_tier_a.prepare_goals
    monkeypatch.setattr(isg_tier_a, 'prepare_goals', lambda goal, *a, **k: captured.append(goal.clone()) or real(goal, *a, **k))
    result = isg_tier_a.run(isg_tier_a.parser().parse_args([str(plain_path), '--frames-per-task', '4', '--device', 'cpu',
                                                            '--batch-size', '8', '--threads', '1', '--goal-mismatch-views', *WRISTS]))
    assert result['goal']['goal_mismatch_views'] == WRISTS and 'evaluation only' in result['probe']['goal_mismatch']
    for own in captured[0::2]:
        for i in range(len(own)):
            (head, _), (left, _), (right, _) = (owner(own[i, view].numpy()) for view in range(3))
            assert left == right != head and EPISODE_TASK[left] == EPISODE_TASK[head]
    monkeypatch.setattr(isg_tier_a, 'prepare_goals', real)
    control = train(parser().parse_args(common(mismatch_root, tmp_path / 'control') + ['--goal-mismatch-views', *WRISTS,
                                                                                      '--max-steps', '1']))
    with pytest.raises(SystemExit, match='applies exactly those'):
        isg_tier_a.run(isg_tier_a.parser().parse_args([str(control), '--device', 'cpu', '--goal-mismatch-views', WRISTS[0]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_three_step_cuda_smoke(mismatch_root, tmp_path):
    output = tmp_path / 'cuda'
    path = train(parser().parse_args(common(mismatch_root, output, device='cuda')
                                     + ['--goal-mismatch-views', *WRISTS, '--max-steps', '3']))
    checkpoint = load_checkpoint(path)
    assert checkpoint['step'] == 3 and checkpoint['adapter_config']['goal_mismatch_views'] == WRISTS
    assert all(torch.isfinite(value).all() for value in checkpoint['model'].values() if value.is_floating_point())
    losses = [json.loads(line).get('loss') for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert len([loss for loss in losses if loss is not None]) == 3 and all(np.isfinite(loss) for loss in losses if loss is not None)
