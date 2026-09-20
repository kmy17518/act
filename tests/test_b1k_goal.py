"""Goal-image conditioning regressions: data contract, early/late fusion invariants, regimes, lifecycle, learnability."""

import copy
import json

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torchvision import transforms

from b1k_dataset import (B1KDataset, CAMERAS, GOAL_OBS_KEYS, GOAL_VIDEO_KEYS, OBS_KEYS, VIDEO_KEYS, StepBatchSampler,
                         preprocess_image, quantize_image)
from b1k_server import PolicyPredictor, Session
from b1k_training import (consume_batch, load_checkpoint, make_policy, optimizer_batches, parser, resolve_regime,
                          train)
from detr.models.backbone import PairedConv2d
from test_b1k import observation, small_model_config, tiny_root  # noqa: F401  (fixture)


NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def goal_config(**overrides):
    config = dict(small_model_config(), state_dim=25, goal_fusion='late', goal_views=['zed_link'])
    config.update(overrides)
    return config


def inputs(batch=2, size=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {'qpos': torch.randn(batch, 25, generator=generator),
            'images': torch.rand(batch, 3, 3, size, size, generator=generator),
            'goal': torch.rand(batch, 1, 3, size, size, generator=generator),
            'actions': torch.randn(batch, 4, 23, generator=generator),
            'pad': torch.tensor([[False, False, True, True], [False, False, False, True]])[:batch]}


def write_video(path, frames, fps=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=fps)
        stream.width = stream.height = 16
        stream.pix_fmt = 'bgr0'
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def add_goal_streams(root):
    """Give the tiny root dedicated goal streams (frames i*10+5) so `goal_source=goal_key` can be exercised."""
    info = json.loads((root / 'meta/info.json').read_text())
    episodes = pq.read_table(root / 'meta/episodes/chunk-042/file-000.parquet')
    for camera in CAMERAS:
        key = GOAL_VIDEO_KEYS[camera]
        write_video(root / info['video_path'].format(video_key=key, chunk_index=42, file_index=0),
                    [np.full((16, 16, 3), i * 10 + 5, np.uint8) for i in range(20)])
        rows = episodes.num_rows
        episodes = episodes.append_column(f'videos/{key}/chunk_index', pa.array([42] * rows))
        episodes = episodes.append_column(f'videos/{key}/file_index', pa.array([0] * rows))
        # episode 42 -> goal frame 3 (value 35), episode 99 -> goal frame 12 (value 125)
        episodes = episodes.append_column(f'videos/{key}/from_timestamp', pa.array([0.3, 1.2]))
        episodes = episodes.append_column(f'videos/{key}/to_timestamp', pa.array([0.9, 1.8]))
    pq.write_table(episodes, root / 'meta/episodes/chunk-042/file-000.parquet')


def test_dataset_goal_table_is_the_episode_last_frame_and_task_id_column(tiny_root):
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), goal_views=['zed_link', 'right_realsense_link'],
                         task_onehot=False)
    dataset.stats = dataset.compute_stats()
    assert dataset.goal_table.shape == (2, 2, 16, 16, 3) and dataset.goal_table.dtype == np.uint8
    for position, ep in enumerate(dataset.episodes):
        last = dataset.raw_sample(ep['episode_index'], ep['length'] - 1)[0]
        for view, camera in enumerate([0, 2]):
            expected = quantize_image(preprocess_image(last[camera], (16, 16)))
            np.testing.assert_array_equal(dataset.goal_table[position, view], expected)
        # tiny_root frames hold value frame_index * 10 + camera; the last observation of episode e is frame e*8+c*2+5
        assert int(dataset.goal_table[position, 0, 0, 0, 0]) == (position * 8 + 5) * 10
        images, qpos, actions, is_pad, goal, task_id = dataset.sample_at(ep['episode_index'], 0)
        assert qpos.shape == (25,) and goal.shape == (2, 16, 16, 3) and goal.dtype == torch.uint8
        assert int(task_id) == ep['task_index'] and torch.equal(goal, torch.from_numpy(dataset.goal_table[position]))
    onehot = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16))
    onehot.stats = dataset.stats
    sample = onehot.sample_at(99, 2)
    assert sample[1].shape == (27,) and sample[4].shape == (0, 16, 16, 3) and int(sample[5]) == 9
    with pytest.raises(ValueError, match='Goal views'):
        B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), goal_views=['zed_link', 'zed_link'])
    with pytest.raises(ValueError, match='goal_source'):
        B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), goal_views=['zed_link'], goal_source='terminal')
    dataset.close()
    onehot.close()


def test_dataset_goal_key_source_reads_the_dedicated_stream(tiny_root):
    with pytest.raises(ValueError, match='no metadata for video streams'):
        B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), goal_views=['zed_link'], goal_source='goal_key')
    add_goal_streams(tiny_root)
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(16, 16), goal_views=['zed_link'], goal_source='goal_key')
    assert [int(dataset.goal_table[p, 0, 0, 0, 0]) for p in range(2)] == [35, 125]
    dataset.close()


def test_goal_free_configuration_is_the_unchanged_base_policy():
    torch.manual_seed(3)
    base = make_policy(small_model_config(), 'cpu')
    torch.manual_seed(3)
    explicit = make_policy(dict(small_model_config(), goal_fusion='none', goal_views=[]), 'cpu')
    assert base.state_dict().keys() == explicit.state_dict().keys()
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], atol=0, rtol=0)
    assert not any(isinstance(module, PairedConv2d) for module in base.modules())
    with pytest.raises(ValueError, match='goal_views require'):
        make_policy(dict(small_model_config(), goal_views=['zed_link']), 'cpu')
    with pytest.raises(ValueError, match='requires goal_views'):
        make_policy(dict(small_model_config(), goal_fusion='late'), 'cpu')
    with pytest.raises(ValueError, match='no goal-image path'):
        base(inputs()['qpos'].new_zeros(2, 27), inputs()['images'], goal=inputs()['goal'])


def base_and_goal_policies(fusion, seed=5, **overrides):
    torch.manual_seed(seed)
    base = make_policy(dict(small_model_config(), state_dim=25), 'cpu')
    torch.manual_seed(seed)
    conditioned = make_policy(goal_config(goal_fusion=fusion, **overrides), 'cpu')
    base.eval()
    conditioned.eval()
    return base, conditioned


def test_early_fusion_stem_reproduces_the_base_stem_at_initialization_and_for_absent_goals():
    base, early = base_and_goal_policies('early')
    stem = early.model.backbones[0][0].body.conv1
    assert isinstance(stem, PairedConv2d) and stem.weight.shape == (64, 3, 7, 7) and stem.goal_weight.shape == (64, 3, 7, 7)
    assert torch.count_nonzero(stem.goal_weight) == 0
    base_state, early_state = base.state_dict(), early.state_dict()
    assert set(early_state) - set(base_state) == {'model.backbones.0.0.body.conv1.goal_weight'}
    for key, value in base_state.items():
        torch.testing.assert_close(value, early_state[key], atol=0, rtol=0)
    x = inputs()
    with torch.no_grad():
        reference = base(x['qpos'], x['images'])
        torch.testing.assert_close(early(x['qpos'], x['images'], goal=x['goal']), reference)
        # the stem is one 6-channel convolution [W_obs, W_goal]: equivalent to conv(obs) + conv_goal(goal)
        stem.goal_weight.normal_()
        paired = torch.cat([x['images'][:, 0], x['goal'][:, 0]], dim=1)
        expected = torch.nn.functional.conv2d(x['images'][:, 0], stem.weight, None, 2, 3) + \
            torch.nn.functional.conv2d(x['goal'][:, 0], stem.goal_weight, None, 2, 3)
        torch.testing.assert_close(stem(paired), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(stem(x['images'][:, 0]), torch.nn.functional.conv2d(x['images'][:, 0], stem.weight, None, 2, 3))
        sensitive = early(x['qpos'], x['images'], goal=x['goal'])
        assert not torch.allclose(sensitive, reference)
        # an absent goal zeroes its channels: exactly the base computation again
        absent = early(x['qpos'], x['images'], goal=x['goal'], goal_valid=torch.zeros(2, 1, dtype=torch.bool))
        torch.testing.assert_close(absent, reference)
        other_goal = early(x['qpos'], x['images'], goal=torch.rand_like(x['goal']), goal_valid=torch.zeros(2, 1, dtype=torch.bool))
        torch.testing.assert_close(other_goal, reference)
    with pytest.raises(ValueError, match='shared_base'):
        make_policy(goal_config(goal_fusion='early', goal_encoder='separate_base'), 'cpu')


def test_early_fusion_camera_batch_matches_per_camera_passes():
    torch.manual_seed(7)
    per_camera = make_policy(goal_config(goal_fusion='early'), 'cpu').eval()
    torch.manual_seed(7)
    batched = make_policy(goal_config(goal_fusion='early', camera_batch=True), 'cpu').eval()
    with torch.no_grad():
        per_camera.model.backbones[0][0].body.conv1.goal_weight.normal_()
        batched.load_state_dict(per_camera.state_dict())
        x = inputs()
        torch.testing.assert_close(batched(x['qpos'], x['images'], goal=x['goal']), per_camera(x['qpos'], x['images'], goal=x['goal']),
                                   atol=1e-5, rtol=1e-5)


def test_late_fusion_appends_goal_tokens_with_identity_and_masks_absent_goals():
    base, late = base_and_goal_policies('late')
    base_state, late_state = base.state_dict(), late.state_dict()
    assert set(late_state) - set(base_state) == {'model.goal_role_embed.weight'}
    assert torch.count_nonzero(late_state['model.goal_role_embed.weight']) == 0
    for key, value in base_state.items():
        torch.testing.assert_close(value, late_state[key], atol=0, rtol=0)
    captured = {}
    original = late.model.transformer.forward

    def spy(src, mask, *args, **kwargs):
        captured['src'], captured['mask'] = src.shape, None if mask is None else mask.clone()
        return original(src, mask, *args, **kwargs)
    late.model.transformer.forward = spy
    x = inputs()
    with torch.no_grad():
        reference = base(x['qpos'], x['images'])
        conditioned = late(x['qpos'], x['images'], goal=x['goal'])
        # 3 camera grids plus 1 goal grid of 1x1 tokens at 32 px (stride 32), no mask when every goal is present
        assert captured['src'] == (2, 32, 1, 4) and captured['mask'] is None
        assert not torch.allclose(conditioned, reference)
        assert not torch.allclose(conditioned, late(x['qpos'], x['images'], goal=torch.rand_like(x['goal'])))
        # masked-out goal tokens: the attention reduces to the base model's, whatever the goal pixels are
        invalid = torch.zeros(2, 1, dtype=torch.bool)
        masked = late(x['qpos'], x['images'], goal=x['goal'], goal_valid=invalid)
        assert captured['mask'].tolist() == [[False, False, False, False, False, True]] * 2
        torch.testing.assert_close(masked, reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(late(x['qpos'], x['images'], goal=torch.rand_like(x['goal']), goal_valid=invalid), masked)
        # per-sample validity: only the invalid row matches the base
        mixed = late(x['qpos'], x['images'], goal=x['goal'], goal_valid=torch.tensor([[True], [False]]))
        torch.testing.assert_close(mixed[1], reference[1], atol=1e-5, rtol=1e-5)
        assert not torch.allclose(mixed[0], reference[0])
    # role embedding off and a separate goal backbone are selectable
    torch.manual_seed(5)
    separate = make_policy(goal_config(goal_fusion='late', goal_encoder='separate_base', goal_role_embedding=False), 'cpu')
    assert separate.model.goal_backbone is not None and separate.model.goal_role_embed is None
    assert any(key.startswith('model.goal_backbone.') for key in separate.state_dict())
    with pytest.raises(ValueError, match='requires goal images'):
        late(x['qpos'], x['images'])


def test_language_on_goal_encoder_false_runs_identity_film():
    embeddings = torch.randn(2, 384)
    outputs = {}
    for on_goal in (False, True):
        torch.manual_seed(11)
        policy = make_policy(goal_config(goal_fusion='late', language_conditioning='mt_act', language_encoder='minilm',
                                         film_init='identity', language_on_goal_encoder=on_goal), 'cpu').eval()
        x = inputs()
        with torch.no_grad():
            outputs[on_goal] = policy(x['qpos'], x['images'], lang_emb=embeddings, goal=x['goal'])
    # zero FiLM generators: modulating the goal pass or skipping it (gamma = beta = 0) is the same computation
    torch.testing.assert_close(outputs[False], outputs[True])
    torch.manual_seed(11)
    policy = make_policy(goal_config(goal_fusion='late', language_conditioning='mt_act', language_encoder='minilm',
                                     film_init='random', language_on_goal_encoder=False), 'cpu').eval()
    with torch.no_grad():
        x = inputs()
        first = policy(x['qpos'], x['images'], lang_emb=embeddings, goal=x['goal'])
        second = policy(x['qpos'], x['images'], lang_emb=embeddings.flip(0), goal=x['goal'])
    assert not torch.allclose(first, second)  # language still reaches the observation pass and the task token


def test_regime_validation_and_defaults():
    def resolve(*flags):
        args = parser().parse_args(['--dataset-path', 'x', '--output-dir', 'y', *flags])
        resolve_regime(args)
        return args

    none = resolve('--regime', 'none')
    assert none.task_onehot is False and none.goal_fusion == 'none' and none.goal_views == [] and none.language_conditioning == 'none'
    image = resolve('--regime', 'image')
    assert image.goal_fusion == 'late' and image.goal_views == ['zed_link'] and image.task_onehot is False
    early = resolve('--regime', 'image', '--goal-fusion', 'early', '--goal-views', 'zed_link', 'left_realsense_link')
    assert early.goal_fusion == 'early' and early.goal_views == ['zed_link', 'left_realsense_link']
    language = resolve('--regime', 'language')
    assert language.language_conditioning == 'mt_act' and language.goal_fusion == 'none'
    both = resolve('--regime', 'image_language', '--language-conditioning', 'clip_film')
    assert both.language_conditioning == 'clip_film' and both.goal_fusion == 'late'
    legacy = resolve()
    assert legacy.task_onehot is True and legacy.regime is None
    assert resolve('--regime', 'none', '--task-onehot').task_onehot is True
    for flags, message in [(['--regime', 'none', '--goal-fusion', 'late'], 'excludes goal images'),
                           (['--regime', 'none', '--language-conditioning', 'mt_act'], 'excludes task-dependent language'),
                           (['--regime', 'image', '--goal-fusion', 'none'], 'requires --goal-fusion'),
                           (['--regime', 'language', '--language-conditioning', 'none'], 'requires --language-conditioning'),
                           (['--regime', 'image', '--language-conditioning', 'clip_film'], 'excludes task-dependent language'),
                           (['--goal-views', 'zed_link'], 'require --goal-fusion'),
                           (['--goal-fusion', 'early', '--goal-encoder', 'separate_base'], 'shared_base'),
                           (['--goal-fusion', 'late', '--language-on-goal-encoder'], 'requires language conditioning'),
                           (['--goal-role-embedding'], 'require --goal-fusion')]:
        with pytest.raises(ValueError, match=message):
            resolve(*flags)


@pytest.mark.parametrize('fusion', ['late', 'early'])
def test_train_resume_serve_goal_conditioned_policy(tiny_root, tmp_path, fusion):
    torch.set_num_threads(1)
    common = ['--dataset-path', str(tiny_root), '--batch-size', '2', '--num-workers', '0', '--torch-threads', '1',
              '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
              '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4', '--no-pretrained-backbone',
              '--save-every', '1', '--dropout', '0.1', '--export-every', '1', '--lr', '1e-3', '--lr-backbone', '1e-3',
              '--regime', 'image', '--goal-fusion', fusion]
    full, split = tmp_path / 'full', tmp_path / 'split'
    expected_path = train(parser().parse_args(common + ['--output-dir', str(full), '--max-steps', '2']))
    first_path = train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '1']))
    first = load_checkpoint(first_path)
    config, adapter = first['model_config'], first['adapter_config']
    assert config['regime'] == 'image' and config['goal_fusion'] == fusion and config['goal_views'] == ['zed_link']
    assert config['state_dim'] == 25 and config['task_onehot'] is False and adapter['task_conditioning'] == 'none'
    assert adapter['goal_source'] == 'episode_last' and adapter['goal_observation_keys'] == [GOAL_OBS_KEYS['zed_link']]
    assert first['conditioning']['regime'] == 'image' and first['conditioning']['goal']['fusion'] == fusion
    assert first['conditioning']['language']['implementation'] == 'none' and first['conditioning']['task_onehot'] is False
    run = json.loads((split / 'run.json').read_text())
    assert run['conditioning']['goal']['views'] == ['zed_link'] and 'source_commit' in run['conditioning']
    for option, value in [('--goal-fusion', 'none' if fusion == 'late' else 'late'), ('--goal-source', 'goal_key'),
                          ('--regime', 'none')]:
        with pytest.raises(ValueError, match='differs from checkpoint|excludes|requires'):
            train(parser().parse_args(common[:-4] + ['--output-dir', str(split), '--resume', str(first_path),
                                                     '--max-steps', '2', option, value]))
    resumed_path = train(parser().parse_args(common + ['--output-dir', str(split), '--resume', str(first_path),
                                                      '--max-steps', '2', '--loader-batch-size', '1']))
    resumed, expected = load_checkpoint(resumed_path), load_checkpoint(expected_path)
    assert resumed['model'].keys() == expected['model'].keys()
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)
    exported = load_checkpoint(split / 'export_queue/eval/step_00000002.pt')
    tiny_root.rename(tmp_path / 'hidden_dataset')
    predictor = PolicyPredictor(exported, 'cpu')
    goal_key = GOAL_OBS_KEYS['zed_link']
    obs = observation(2, np.array([7, 9]))
    with pytest.raises(ValueError, match='needs goal'):
        Session(predictor, exported, action_horizon=2).act(obs)
    obs[goal_key] = np.full((2, 16, 16, 3), 40, dtype=np.uint8)
    session = Session(predictor, exported, action_horizon=2)
    actions = session.act(obs)
    assert actions.shape == (2, 23) and np.isfinite(actions).all()
    reference = Session(PolicyPredictor(resumed, 'cpu'), resumed, action_horizon=2).act(obs)
    np.testing.assert_array_equal(actions, reference)
    # image regime: the task id selects nothing in the network, so it cannot change the action
    swapped = dict(obs, task_id=np.array([9, 7]))
    np.testing.assert_array_equal(Session(predictor, exported, action_horizon=2).act(swapped), actions)
    # ... whereas the goal image does (trained weights: the zero-initialised early stem has moved after two steps)
    other = dict(obs)
    other[goal_key] = np.full((2, 16, 16, 3), 200, dtype=np.uint8)
    assert not np.array_equal(Session(predictor, exported, action_horizon=2).act(other), actions)
    # fixed goal images given at server start stand in for missing request goals
    fixed = Session(predictor, exported, action_horizon=2, fixed_goals={'zed_link': np.full((16, 16, 3), 40, np.uint8)})
    np.testing.assert_array_equal(fixed.act(observation(2, np.array([7, 9]))), actions)
    assert all(key not in OBS_KEYS for key in [goal_key])


def two_goal_root(tmp_path):
    """Two episodes with identical observations/states, different dedicated goal streams and opposite actions."""
    root = tmp_path / 'two_goal'
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    info = {'codebase_version': 'v3.0', 'fps': 10,
            'features': {'action': {'shape': [23]}, 'observation.state': {'shape': [61]}},
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mkv'}
    (root / 'meta/info.json').write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist([{'task_index': 0, 'task': 'reach'}]), root / 'meta/tasks.parquet')
    rng = np.random.default_rng(0)
    for c, camera in enumerate(CAMERAS):
        frames = [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(8)]  # shared observation frames
        write_video(root / info['video_path'].format(video_key=VIDEO_KEYS[c], chunk_index=0, file_index=0), frames)
        # two textured goal images (a random-init ResNet with identity frozen BN maps constant images to dead features)
        goal_a, goal_b = (rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(2))
        write_video(root / info['video_path'].format(video_key=GOAL_VIDEO_KEYS[camera], chunk_index=0, file_index=0),
                    [goal_a] * 4 + [goal_b] * 4)
    rows, episodes = [], []
    for e in range(2):
        ep = {'episode_index': e, 'task_index': 0, 'length': 6, 'dataset_from_index': 6 * e, 'dataset_to_index': 6 * e + 6,
              'data/chunk_index': 0, 'data/file_index': 0}
        for c, camera in enumerate(CAMERAS):
            ep.update({f'videos/{VIDEO_KEYS[c]}/chunk_index': 0, f'videos/{VIDEO_KEYS[c]}/file_index': 0,
                       f'videos/{VIDEO_KEYS[c]}/from_timestamp': 0.0, f'videos/{VIDEO_KEYS[c]}/to_timestamp': 0.6,
                       f'videos/{GOAL_VIDEO_KEYS[camera]}/chunk_index': 0, f'videos/{GOAL_VIDEO_KEYS[camera]}/file_index': 0,
                       f'videos/{GOAL_VIDEO_KEYS[camera]}/from_timestamp': 0.4 * e, f'videos/{GOAL_VIDEO_KEYS[camera]}/to_timestamp': 0.4 * e + 0.4})
        episodes.append(ep)
        for frame in range(6):
            state = np.arange(61, dtype=np.float32) * .01 + frame * .1
            action = np.sin(np.arange(23, dtype=np.float32) + frame * .3) * (1.0 if e == 0 else -1.0)
            rows.append({'episode_index': e, 'task_index': 0, 'index': 6 * e + frame, 'frame_index': frame,
                         'timestamp': frame / 10, 'observation.state': state.tolist(), 'action': action.tolist()})
    pq.write_table(pa.Table.from_pylist(rows), root / 'data/chunk-000/file-000.parquet', row_group_size=6)
    pq.write_table(pa.Table.from_pylist(episodes), root / 'meta/episodes/chunk-000/file-000.parquet')
    return root


@pytest.mark.parametrize('fusion', ['late', 'early'])
def test_small_policy_learns_two_goals_from_the_same_start(tmp_path, fusion):
    """Acceptance check: identical observations, two reachable goals, opposite actions; inference uses the zero latent."""
    torch.set_num_threads(2)
    root = two_goal_root(tmp_path)
    dataset = B1KDataset(root, chunk_size=4, image_size=(16, 16), goal_views=['zed_link'], goal_source='goal_key',
                         task_onehot=False)
    dataset.stats = dataset.compute_stats()
    torch.manual_seed(0)
    # KL weight 10 (the B1K recipe): the style latent must stay close to the prior, so the decoder has to read the
    # goal to tell the two action families apart (inference uses the zero latent). The zero-initialised early stem
    # needs a larger backbone learning rate than the late tokens to become goal-sensitive within 150 steps; the
    # policy separates the goals from step ~100 (early) / ~75 (late) at these rates on this synthetic pair.
    policy = make_policy(goal_config(goal_fusion=fusion, lr=1e-3, lr_backbone=2e-3 if fusion == 'early' else 3e-4,
                                     kl_weight=10.0), 'cpu')
    optimizer = policy.configure_optimizers()
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=StepBatchSampler(dataset, 8, 0, 150, 1))
    policy.train()
    for batch in optimizer_batches(loader, 8, None):
        images, qpos, actions, is_pad, goal, task_id, _ = consume_batch(batch)
        optimizer.zero_grad()
        policy(qpos, images, actions, is_pad, goal=goal)['loss'].backward()
        optimizer.step()
    policy.eval()
    samples = [dataset.sample_at(e, 1) for e in range(2)]
    images = consume_batch((samples[0][0][None], samples[0][1][None], samples[0][2][None], samples[0][3][None],
                            samples[0][4][None], samples[0][5][None], {}))[0]
    targets = {e: samples[e][2][:4] for e in range(2)}
    with torch.no_grad():
        predictions = {e: policy(samples[0][1][None], images, goal=consume_batch(
            (samples[e][0][None], samples[e][1][None], samples[e][2][None], samples[e][3][None], samples[e][4][None],
             samples[e][5][None], {}))[4])[0] for e in range(2)}
    assert not torch.allclose(predictions[0], predictions[1])
    for e in range(2):
        own = (predictions[e] - targets[e]).abs().mean()
        other = (predictions[e] - targets[1 - e]).abs().mean()
        assert own < other, f'goal {e} ({fusion}): own target L1 {own:.3f} vs other {other:.3f}'
    dataset.close()
