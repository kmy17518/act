"""Goal-mechanism options of the ISG goal-conditioning study: paired-stem and goal-tag initializations, pooled goal
tokens, the frozen goal encoder and the goal learning-rate group."""

import pytest
import torch

from b1k_training import load_checkpoint, make_policy, parser, resolve_regime, train
from detr.main import is_goal_parameter
from test_b1k import small_model_config, tiny_root  # noqa: F401  (fixture)
from test_b1k_goal import NORMALIZE, base_and_goal_policies, goal_config, inputs


def head_goal_swapped(x):
    """(images, goal) with the head image and the goal image trading places."""
    images = x['images'].clone()
    images[:, 0] = x['goal'][:, 0]
    return images, x['images'][:, :1].clone()


@pytest.mark.parametrize('init', ['zero', 'random', 'copy:0.5', 'copy:1'])
def test_goal_stem_inits_keep_the_base_parameters_and_the_absent_goal(init):
    base, early = base_and_goal_policies('early', goal_stem_init=init)
    stem = early.model.backbones[0][0].body.conv1
    early_state = early.state_dict()
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, early_state[key], atol=0, rtol=0)
    if init == 'zero':
        assert torch.count_nonzero(stem.goal_weight) == 0
    elif init == 'random':
        bound = 1 / (3 * 7 * 7) ** 0.5  # nn.Conv2d's default for the goal half's own fan-in
        assert stem.goal_weight.abs().max() <= bound
        assert abs(stem.goal_weight.std().item() - bound / 3 ** 0.5) < 0.1 * bound / 3 ** 0.5
    else:
        torch.testing.assert_close(stem.goal_weight, float(init[5:]) * stem.weight, atol=0, rtol=0)
    x = inputs()
    images, goal = head_goal_swapped(x)
    with torch.no_grad():
        reference = base(x['qpos'], x['images'])
        absent = early(x['qpos'], x['images'], goal=x['goal'], goal_valid=torch.zeros(2, 1, dtype=torch.bool))
        own = early(x['qpos'], x['images'], goal=x['goal'])
        swapped = early(x['qpos'], images, goal=goal)
    torch.testing.assert_close(absent, reference)
    if init == 'zero':
        torch.testing.assert_close(own, reference)
    elif init == 'copy:1':
        # W * current + W * goal is symmetric: a plain copy cannot tell which image is the goal
        assert not torch.allclose(own, reference)
        torch.testing.assert_close(swapped, own, atol=1e-5, rtol=1e-5)
    else:
        assert not torch.allclose(own, reference)
        assert not torch.allclose(swapped, own, atol=1e-4)


def test_goal_tag_start_decides_whether_late_fusion_can_tell_the_goal_from_the_head_image():
    x = inputs()
    images, goal = head_goal_swapped(x)
    swaps = {}
    for init in ('zero', 'normal:0.5'):
        base, late = base_and_goal_policies('late', goal_tag_init=init)
        late_state = late.state_dict()
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, late_state[key], atol=0, rtol=0)
        tag = late.model.goal_role_embed.weight
        assert torch.count_nonzero(tag) == 0 if init == 'zero' else 0.3 < tag.std().item() < 0.7
        with torch.no_grad():
            swaps[init] = (late(x['qpos'], images, goal=goal) - late(x['qpos'], x['images'], goal=x['goal'])).abs().max()
    # same token contents at the same positions, permuted: attention cannot see the swap without a tag
    assert swaps['zero'] < 1e-5 and swaps['normal:0.5'] > 1e-3


def test_pooled_goal_tokens_are_one_mean_token_per_view_and_mask_exactly():
    base, late = base_and_goal_policies('late', goal_tokens='pooled')
    late_state = late.state_dict()
    assert set(late_state) - set(base.state_dict()) == {'model.goal_role_embed.weight'}
    captured = {}
    original = late.model.transformer.forward

    def spy(src, mask, *args, **kwargs):
        memory = kwargs.get('memory_tokens')
        captured.update(src=src.shape, mask=None if mask is None else mask.clone(),
                        memory=None if memory is None else memory.clone())
        return original(src, mask, *args, **kwargs)
    late.model.transformer.forward = spy
    x = inputs(size=64)  # stride 32: a 2x2 grid per image
    hidden = late.model.transformer.d_model
    with torch.no_grad():
        reference = base(x['qpos'], x['images'])
        conditioned = late(x['qpos'], x['images'], goal=x['goal'])
        assert captured['src'] == (2, hidden, 2, 6) and captured['memory'].shape == (1, 2, hidden)
        assert captured['mask'] is None
        grid = late.model.input_proj(late.model.backbones[0](NORMALIZE(x['goal'][:, 0]))[0][0])
        torch.testing.assert_close(captured['memory'][0], grid.mean(dim=(2, 3)), atol=1e-5, rtol=1e-5)
        assert not torch.allclose(conditioned, reference)
        assert not torch.allclose(conditioned, late(x['qpos'], x['images'], goal=torch.rand_like(x['goal'])))
        masked = late(x['qpos'], x['images'], goal=x['goal'], goal_valid=torch.zeros(2, 1, dtype=torch.bool))
        assert captured['mask'].tolist() == [[False] * (2 + 12) + [True]] * 2
        torch.testing.assert_close(masked, reference, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match='late fusion only'):
        make_policy(goal_config(goal_fusion='early', goal_tokens='pooled'), 'cpu')


def test_frozen_goal_encoder_is_the_initial_backbone_and_never_trains():
    torch.manual_seed(5)
    policy = make_policy(goal_config(goal_fusion='late', goal_encoder='frozen', lr=1e-2, lr_backbone=1e-2), 'cpu')
    frozen, backbone = policy.model.goal_backbone, policy.model.backbones[0]
    assert frozen is not None and not any(p.requires_grad for p in frozen.parameters())
    for (key, value), (_, copy) in zip(backbone.state_dict().items(), frozen.state_dict().items()):
        torch.testing.assert_close(value, copy, atol=0, rtol=0)
    optimizer = policy.configure_optimizers()
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert not any(id(p) in optimized for p in frozen.parameters())
    frozen_before = {k: v.clone() for k, v in frozen.state_dict().items()}
    backbone_before = {k: v.clone() for k, v in backbone.state_dict().items()}
    policy.train()
    assert not frozen.training and backbone.training
    x = inputs()
    policy(x['qpos'], x['images'], x['actions'], x['pad'], goal=x['goal'])['loss'].backward()
    assert all(p.grad is None for p in frozen.parameters())
    optimizer.step()
    for key, value in frozen.state_dict().items():
        torch.testing.assert_close(value, frozen_before[key], atol=0, rtol=0)
    assert any(not torch.equal(value, backbone_before[key]) for key, value in backbone.state_dict().items())
    policy.eval()
    with torch.no_grad():  # the goal still reaches the actions through the trainable projection
        assert not torch.allclose(policy(x['qpos'], x['images'], goal=x['goal']),
                                  policy(x['qpos'], x['images'], goal=torch.rand_like(x['goal'])))
    with pytest.raises(ValueError, match='frozen goal encoder cannot be language-modulated'):
        make_policy(goal_config(goal_fusion='late', goal_encoder='frozen', language_conditioning='mt_act',
                                language_encoder='minilm', film_init='identity', language_on_goal_encoder=True), 'cpu')


@pytest.mark.parametrize('fusion,goal_name', [('late', 'model.goal_role_embed.weight'),
                                              ('early', 'model.backbones.0.0.body.conv1.goal_weight')])
def test_lr_goal_moves_only_the_goal_parameters_into_their_own_group(fusion, goal_name):
    torch.manual_seed(5)
    assert len(make_policy(goal_config(goal_fusion=fusion), 'cpu').configure_optimizers().param_groups) == 2
    torch.manual_seed(5)
    policy = make_policy(goal_config(goal_fusion=fusion, lr_goal=1e-3), 'cpu')
    groups = policy.configure_optimizers().param_groups
    names = {id(p): name for name, p in policy.named_parameters()}
    assert len(groups) == 3 and groups[2]['lr'] == 1e-3
    assert [names[id(p)] for p in groups[2]['params']] == [goal_name]
    assert not any(is_goal_parameter(names[id(p)]) for group in groups[:2] for p in group['params'])
    assert sum(len(group['params']) for group in groups) == sum(p.requires_grad for p in policy.parameters())


def test_goal_mechanism_flags_are_validated_and_canonicalized():
    def resolve(*flags, regime='image'):
        args = parser().parse_args(['--dataset-path', 'x', '--output-dir', 'y', '--regime', regime, *flags])
        resolve_regime(args)
        return args

    early = resolve('--goal-fusion', 'early', '--goal-stem-init', 'copy:0.50', '--lr-goal', '1e-4')
    assert early.goal_stem_init == 'copy:0.5' and early.lr_goal == 1e-4 and early.goal_tag_init == 'zero'
    assert resolve('--goal-tag-init', 'normal:0.50', '--goal-tokens', 'pooled').goal_fusion == 'late'
    late = resolve('--goal-fusion', 'late', '--goal-tag-init', 'normal:0.50', '--goal-tokens', 'pooled',
                   '--goal-encoder', 'frozen')
    assert late.goal_tag_init == 'normal:0.5' and late.goal_tokens == 'pooled' and late.goal_encoder == 'frozen'
    for flags, message in [(['--goal-fusion', 'early', '--goal-tokens', 'pooled'], 'apply to --goal-fusion late'),
                           (['--goal-fusion', 'early', '--goal-tag-init', 'normal:1'], 'apply to --goal-fusion late'),
                           (['--goal-stem-init', 'random'], 'applies to --goal-fusion early'),
                           (['--goal-tag-init', 'normal:0'], 'goal tag init'),
                           (['--goal-fusion', 'early', '--goal-stem-init', 'copy:x'], 'goal stem init'),
                           (['--goal-fusion', 'late', '--goal-tag-init', 'normal:1', '--no-goal-role-embedding'],
                            'requires the goal role'),
                           (['--lr-goal', '0'], 'positive')]:
        with pytest.raises(ValueError, match=message):
            resolve(*flags)
    with pytest.raises(ValueError, match='--lr-goal requires'):
        resolve('--lr-goal', '1e-4', regime='none')


@pytest.mark.parametrize('fusion,flags', [
    ('late', ['--goal-tokens', 'pooled', '--goal-encoder', 'frozen', '--goal-tag-init', 'normal:0.5', '--lr-goal', '1e-2']),
    ('early', ['--goal-stem-init', 'copy:0.5', '--lr-goal', '1e-2'])])
def test_goal_mechanisms_are_recorded_and_resume_exactly(tiny_root, tmp_path, fusion, flags):
    torch.set_num_threads(1)
    common = ['--dataset-path', str(tiny_root), '--batch-size', '2', '--num-workers', '0', '--torch-threads', '1',
              '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
              '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
              '--no-pretrained-backbone', '--save-every', '1', '--lr', '1e-3', '--lr-backbone', '1e-3',
              '--regime', 'image', '--goal-fusion', fusion, *flags]
    full, split = tmp_path / 'full', tmp_path / 'split'
    expected = load_checkpoint(train(parser().parse_args(common + ['--output-dir', str(full), '--max-steps', '2'])))
    first_path = train(parser().parse_args(common + ['--output-dir', str(split), '--max-steps', '1']))
    first = load_checkpoint(first_path)
    config, goal = first['model_config'], first['conditioning']['goal']
    assert config['lr_goal'] == 1e-2 and goal['lr_goal'] == 1e-2
    if fusion == 'late':
        assert (config['goal_tokens'], config['goal_encoder'], config['goal_tag_init']) == ('pooled', 'frozen', 'normal:0.5')
        assert (goal['tokens'], goal['tag_init'], goal['stem_init']) == ('pooled', 'normal:0.5', None)
        assert len(first['optimizer']['param_groups']) == 3
    else:
        assert config['goal_stem_init'] == 'copy:0.5' and (goal['stem_init'], goal['tokens']) == ('copy:0.5', None)
    with pytest.raises(ValueError, match='differs from checkpoint'):
        train(parser().parse_args(common + ['--output-dir', str(split), '--resume', str(first_path), '--max-steps', '2',
                                            '--lr-goal', '1e-3']))
    resumed = load_checkpoint(train(parser().parse_args(common + ['--output-dir', str(split), '--resume',
                                                                  str(first_path), '--max-steps', '2'])))
    assert resumed['model'].keys() == expected['model'].keys()
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)
