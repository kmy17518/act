"""Wave-2 goal mechanisms of the ISG goal-conditioning study: goal-token position codes, difference tokens, where the
goal enters (encoder, decoder only, action queries), and the paired stem's learned gain and difference channel."""

import pytest
import torch

from b1k_training import load_checkpoint, make_policy, parser, resolve_regime, train
from detr.main import is_goal_parameter
from test_b1k import small_model_config, tiny_root  # noqa: F401  (fixture)
from test_b1k_goal import base_and_goal_policies, goal_config, inputs
from test_b1k_goal_mechanisms import head_goal_swapped


def spy_transformer(policy):
    captured = {}
    original = policy.model.transformer.forward

    def spy(src, mask, *args, **kwargs):
        captured.update(src=src.clone(), mask=None if mask is None else mask.clone(),
                        **{key: None if kwargs.get(key) is None else kwargs[key].clone()
                           for key in ('memory_tokens', 'decoder_tokens', 'decoder_mask', 'query_add')})
        return original(src, mask, *args, **kwargs)
    policy.model.transformer.forward = spy
    return captured


def assert_masked_goal_is_the_base(base, policy, x):
    with torch.no_grad():
        reference = base(x['qpos'], x['images'])
        invalid = torch.zeros(len(x['qpos']), 1, dtype=torch.bool)
        masked = policy(x['qpos'], x['images'], goal=x['goal'], goal_valid=invalid)
        torch.testing.assert_close(masked, reference, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(policy(x['qpos'], x['images'], goal=torch.rand_like(x['goal']), goal_valid=invalid),
                                   masked)
        assert not torch.allclose(policy(x['qpos'], x['images'], goal=x['goal']), reference)


@pytest.mark.parametrize('overrides', [dict(goal_pos='none'), dict(goal_content='diff'), dict(goal_entry='decoder'),
                                       dict(goal_entry='decoder', goal_tokens='pooled'),
                                       dict(goal_entry='queries', goal_tokens='pooled', goal_role_embedding=False)])
def test_late_wave2_options_keep_the_base_parameters_and_the_exact_absent_goal(overrides):
    base, late = base_and_goal_policies('late', **overrides)
    late_state = late.state_dict()
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, late_state[key], atol=0, rtol=0)
    assert_masked_goal_is_the_base(base, late, inputs(size=64))


def test_goal_tokens_without_position_codes_tell_the_goal_from_the_head_image_at_a_zero_tag():
    x = inputs(size=64)
    images, goal = head_goal_swapped(x)
    swaps = {}
    for positions in ('sine', 'none'):
        _, late = base_and_goal_policies('late', goal_pos=positions)
        with torch.no_grad():
            swaps[positions] = (late(x['qpos'], images, goal=goal) - late(x['qpos'], x['images'], goal=x['goal'])).abs().max()
    assert swaps['sine'] < 1e-5 and swaps['none'] > 1e-3


def test_difference_tokens_are_goal_minus_current_features_and_vanish_for_the_current_image():
    _, late = base_and_goal_policies('late', goal_content='diff')
    captured = spy_transformer(late)
    x = inputs(size=64)
    with torch.no_grad():
        late(x['qpos'], x['images'], goal=x['images'][:, :1].clone())  # goal == the head camera's current image
    hidden = late.model.transformer.d_model
    assert captured['src'].shape == (2, hidden, 2, 8)
    torch.testing.assert_close(captured['src'][..., 6:], torch.zeros_like(captured['src'][..., 6:]), atol=1e-5, rtol=0)


def test_decoder_entry_keeps_the_goal_out_of_the_encoder():
    _, late = base_and_goal_policies('late', goal_entry='decoder')
    captured = spy_transformer(late)
    x = inputs(size=64)
    hidden = late.model.transformer.d_model
    encoded = []
    original = late.model.transformer.encoder.forward
    late.model.transformer.encoder.forward = lambda src, **kwargs: encoded.append(src.clone()) or original(src, **kwargs)
    with torch.no_grad():
        late(x['qpos'], x['images'], goal=x['goal'])
        late(x['qpos'], x['images'], goal=torch.rand_like(x['goal']))
    assert captured['src'].shape == (2, hidden, 2, 6) and captured['decoder_tokens'].shape == (4, 2, hidden)
    torch.testing.assert_close(encoded[0], encoded[1], atol=0, rtol=0)  # the encoder never sees the goal
    with torch.no_grad():
        late(x['qpos'], x['images'], goal=x['goal'], goal_valid=torch.tensor([[True], [False]]))
    assert captured['decoder_mask'].tolist() == [[False] * 4, [True] * 4]


def test_query_entry_adds_the_pooled_goal_to_every_action_query():
    _, late = base_and_goal_policies('late', goal_entry='queries', goal_tokens='pooled', goal_role_embedding=False)
    captured = spy_transformer(late)
    x = inputs(size=64)
    with torch.no_grad():
        late(x['qpos'], x['images'], goal=x['goal'])
    hidden = late.model.transformer.d_model
    assert captured['src'].shape == (2, hidden, 2, 6) and captured['memory_tokens'] is None
    assert captured['query_add'].shape == (2, hidden) and captured['decoder_tokens'] is None
    for overrides in (dict(goal_tokens='grid', goal_role_embedding=False), dict(goal_tokens='pooled')):
        with pytest.raises(ValueError, match="goal_entry='queries'"):
            make_policy(goal_config(goal_fusion='late', goal_entry='queries', **overrides), 'cpu')


def test_stem_gain_starts_as_the_scaled_copy_and_is_a_goal_parameter():
    torch.manual_seed(5)
    gained = make_policy(goal_config(goal_fusion='early', goal_stem_init='copy:1', goal_stem_gain=0.5, lr_goal=1e-3), 'cpu')
    torch.manual_seed(5)
    scaled = make_policy(goal_config(goal_fusion='early', goal_stem_init='copy:0.5'), 'cpu')
    gained.eval(), scaled.eval()
    stem = gained.model.backbones[0][0].body.conv1
    assert stem.goal_gain.shape == (64, 1, 1, 1) and torch.all(stem.goal_gain == 0.5)
    x = inputs()
    with torch.no_grad():
        torch.testing.assert_close(gained(x['qpos'], x['images'], goal=x['goal']),
                                   scaled(x['qpos'], x['images'], goal=x['goal']), atol=1e-5, rtol=1e-5)
    names = {id(p): n for n, p in gained.named_parameters()}
    goal_group = [names[id(p)] for p in gained.configure_optimizers().param_groups[2]['params']]
    assert goal_group == ['model.backbones.0.0.body.conv1.goal_weight', 'model.backbones.0.0.body.conv1.goal_gain']


def test_stem_difference_channel_starts_neutral_computes_goal_minus_current_and_keeps_the_absent_goal():
    base, plain = base_and_goal_policies('early', goal_stem_init='copy:0.5')
    _, diff = base_and_goal_policies('early', goal_stem_init='copy:0.5', goal_stem_diff=True)
    stem = diff.model.backbones[0][0].body.conv1
    assert stem.paired_channels == 9 and torch.count_nonzero(stem.goal_diff_weight) == 0
    assert is_goal_parameter('model.backbones.0.0.body.conv1.goal_diff_weight')
    x = inputs()
    with torch.no_grad():
        torch.testing.assert_close(diff(x['qpos'], x['images'], goal=x['goal']),
                                   plain(x['qpos'], x['images'], goal=x['goal']), atol=1e-5, rtol=1e-5)
        stem.goal_diff_weight.normal_(0, 0.05)
        current, goal = x['images'][:, 0], x['goal'][:, 0]
        expected = sum(torch.nn.functional.conv2d(image, weight, None, 2, 3) for image, weight in
                       [(current, stem.weight), (goal, stem.goal_weight), (goal - current, stem.goal_diff_weight)])
        torch.testing.assert_close(stem(torch.cat([current, goal, goal - current], dim=1)), expected, atol=1e-5, rtol=1e-5)
    assert_masked_goal_is_the_base(base, diff, x)
    torch.manual_seed(7)
    per_camera = make_policy(goal_config(goal_fusion='early', goal_stem_diff=True), 'cpu').eval()
    torch.manual_seed(7)
    batched = make_policy(goal_config(goal_fusion='early', goal_stem_diff=True, camera_batch=True), 'cpu').eval()
    with torch.no_grad():
        per_camera.model.backbones[0][0].body.conv1.goal_diff_weight.normal_()
        batched.load_state_dict(per_camera.state_dict())
        torch.testing.assert_close(batched(x['qpos'], x['images'], goal=x['goal']),
                                   per_camera(x['qpos'], x['images'], goal=x['goal']), atol=1e-5, rtol=1e-5)


def test_wave2_flags_are_validated():
    def resolve(*flags):
        args = parser().parse_args(['--dataset-path', 'x', '--output-dir', 'y', '--regime', 'image', *flags])
        resolve_regime(args)
        return args

    queries = resolve('--goal-fusion', 'late', '--goal-entry', 'queries', '--goal-tokens', 'pooled', '--no-goal-role-embedding')
    assert queries.goal_entry == 'queries'
    early = resolve('--goal-fusion', 'early', '--goal-stem-gain', '0.5', '--goal-stem-diff')
    assert early.goal_stem_gain == 0.5 and early.goal_stem_diff is True
    for flags, message in [(['--goal-fusion', 'early', '--goal-pos', 'none'], 'apply to --goal-fusion late'),
                           (['--goal-fusion', 'early', '--goal-entry', 'decoder'], 'apply to --goal-fusion late'),
                           (['--goal-fusion', 'late', '--goal-stem-gain', '0.5'], 'apply to --goal-fusion early'),
                           (['--goal-fusion', 'late', '--goal-stem-diff'], 'apply to --goal-fusion early'),
                           (['--goal-fusion', 'late', '--goal-entry', 'queries'], 'requires --goal-tokens pooled')]:
        with pytest.raises(ValueError, match=message):
            resolve(*flags)


@pytest.mark.parametrize('fusion,flags', [
    ('late', ['--goal-pos', 'none', '--goal-content', 'diff', '--goal-entry', 'decoder', '--lr-goal', '1e-2']),
    ('early', ['--goal-stem-init', 'copy:1', '--goal-stem-gain', '0.5', '--goal-stem-diff', '--lr-goal', '1e-2'])])
def test_wave2_options_are_recorded_and_resume_exactly(tiny_root, tmp_path, fusion, flags):
    torch.set_num_threads(1)
    common = ['--dataset-path', str(tiny_root), '--batch-size', '2', '--num-workers', '0', '--torch-threads', '1',
              '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
              '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
              '--no-pretrained-backbone', '--save-every', '1', '--lr', '1e-3', '--lr-backbone', '1e-3',
              '--regime', 'image', '--goal-fusion', fusion, *flags]
    expected = load_checkpoint(train(parser().parse_args(common + ['--output-dir', str(tmp_path / 'full'),
                                                                   '--max-steps', '2'])))
    first_path = train(parser().parse_args(common + ['--output-dir', str(tmp_path / 'split'), '--max-steps', '1']))
    goal = load_checkpoint(first_path)['conditioning']['goal']
    if fusion == 'late':
        assert (goal['positions'], goal['content'], goal['entry'], goal['stem_gain']) == ('none', 'diff', 'decoder', None)
    else:
        assert (goal['stem_gain'], goal['stem_diff'], goal['entry']) == (0.5, True, None)
    resumed = load_checkpoint(train(parser().parse_args(common + ['--output-dir', str(tmp_path / 'split'), '--resume',
                                                                  str(first_path), '--max-steps', '2'])))
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)
