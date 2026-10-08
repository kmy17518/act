"""--goal-detach-views: the chosen goal views' paired-stem passes do not train the goal half of the stem; the forward
pass and every other gradient are those of the run without it."""

import types

import pytest
import torch

from b1k_training import compile_policy, load_checkpoint, make_policy, parser, train
from detr.models.backbone import PairedConv2d
from test_b1k import small_model_config, tiny_root  # noqa: F401  (fixture)

VIEWS = ['zed_link', 'left_realsense_link', 'right_realsense_link']
WRISTS = VIEWS[1:]
GOAL_PARAMETERS = ('goal_weight', 'goal_gain', 'goal_diff_weight')


def policy(detach, device='cpu', seed=5, stem=None, **options):
    torch.manual_seed(seed)
    config = dict(small_model_config(), state_dim=25, goal_fusion='early', goal_views=VIEWS, goal_stem_init='copy:0.5',
                  lr_goal=1e-3, **(stem or {}), **({'goal_detach_views': WRISTS} if detach else {}))
    return make_policy(config, device, **options)


def batch(device='cpu', size=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    values = {'qpos': torch.randn(2, 25, generator=generator), 'images': torch.rand(2, 3, 3, size, size, generator=generator),
              'goal': torch.rand(2, 3, 3, size, size, generator=generator), 'actions': torch.randn(2, 4, 23, generator=generator),
              'pad': torch.tensor([[False, False, True, True], [False, False, False, True]])}
    return {key: value.to(device) for key, value in values.items()}


def gradients(model, x, seed=11):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)  # the CVAE latent sample
    model(x['qpos'], x['images'], x['actions'], x['pad'], goal=x['goal'])['loss'].backward()
    return {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}


def is_goal_half(name):
    return any(name.endswith(f'conv1.{key}') for key in GOAL_PARAMETERS)


@pytest.mark.parametrize('stem', [{}, {'goal_stem_gain': 0.5, 'goal_stem_diff': True}])
def test_forward_is_identical_and_only_the_goal_half_loses_the_wrist_gradients(stem):
    plain, detached = policy(False, stem=stem), policy(True, stem=stem)
    assert plain.state_dict().keys() == detached.state_dict().keys()
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(detached.state_dict()[key], value, atol=0, rtol=0)
    x = batch()
    plain.eval(), detached.eval()
    with torch.no_grad():
        assert torch.equal(plain(x['qpos'], x['images'], goal=x['goal']), detached(x['qpos'], x['images'], goal=x['goal']))
    torch.manual_seed(11)
    plain.train()
    loss_plain = plain(x['qpos'], x['images'], x['actions'], x['pad'], goal=x['goal'])['loss']
    torch.manual_seed(11)
    detached.train()
    assert torch.equal(loss_plain, detached(x['qpos'], x['images'], x['actions'], x['pad'], goal=x['goal'])['loss'])
    full, cut = gradients(plain, x), gradients(detached, x)
    assert full.keys() == cut.keys()
    for name in full:
        if not is_goal_half(name):  # observation half of the stem and every later layer: all three cameras, unchanged
            torch.testing.assert_close(cut[name], full[name], atol=0, rtol=0)
    goal = [name for name in full if is_goal_half(name)]
    assert goal and all(not torch.allclose(cut[name], full[name]) for name in goal)


def test_goal_weight_gradient_is_the_head_pairs_alone(monkeypatch):
    """Per-pass decomposition: give every paired pass its own leaf copy of the goal weight; the detached run's
    goal_weight gradient equals the head pass's copy, the plain run's the sum of all three."""
    plain, detached = policy(False), policy(True)
    x = batch()
    full, cut = gradients(plain, x)['model.backbones.0.0.body.conv1.goal_weight'], \
        gradients(detached, x)['model.backbones.0.0.body.conv1.goal_weight']
    stem = plain.model.backbones[0][0].body.conv1
    copies = []

    def paired_weight(self):
        copy = self.goal_weight.detach().clone().requires_grad_(True)
        copies.append(copy)
        return torch.cat([self.weight, copy.to(self.weight.dtype)], dim=1)

    monkeypatch.setattr(stem, 'paired_weight', types.MethodType(paired_weight, stem))
    gradients(plain, x)
    assert len(copies) == 3 and all(c.grad is not None and c.grad.abs().sum() > 0 for c in copies)  # head, left, right
    torch.testing.assert_close(cut, copies[0].grad, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(full, sum(c.grad for c in copies), atol=1e-6, rtol=1e-5)


def test_unsupported_configurations_are_refused():
    base = dict(small_model_config(), state_dim=25, goal_views=VIEWS, goal_detach_views=WRISTS)
    for options in ({'goal_fusion': 'late'}, {'goal_fusion': 'early', 'goal_fusion_depth': 1},
                    {'goal_fusion': 'early', 'goal_views': ['zed_link']}):
        with pytest.raises(ValueError, match='goal_detach_views'):
            make_policy(dict(base, **options), 'cpu')
    with pytest.raises(ValueError, match='goal_detach_views'):
        make_policy(dict(base, goal_fusion='early', camera_batch=True), 'cpu')
    plain = policy(False)
    assert plain.model.goal_detach_views == () and plain.model.backbones[0][0].body.conv1.detach_goal is False


def common(root, output, device='cpu'):
    return ['--dataset-path', str(root), '--batch-size', '2', '--num-workers', '0', '--torch-threads', '1',
            '--device', device, '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
            '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
            '--no-pretrained-backbone', '--save-every', '1', '--export-every', '1', '--lr', '1e-3', '--lr-backbone', '1e-3',
            '--regime', 'image', '--goal-fusion', 'early', '--goal-stem-init', 'copy:0.5', '--lr-goal', '1e-3',
            '--goal-views', *VIEWS, '--output-dir', str(output)]


def test_option_is_saved_kept_on_resume_and_rejected_when_it_differs(tiny_root, tmp_path):
    torch.set_num_threads(1)
    output = tmp_path / 'detach'
    first_path = train(parser().parse_args(common(tiny_root, output) + ['--goal-detach-views', *WRISTS, '--max-steps', '1']))
    first = load_checkpoint(first_path)
    assert first['model_config']['goal_detach_views'] == WRISTS and first['conditioning']['goal']['detach_views'] == WRISTS
    resume = common(tiny_root, output) + ['--resume', str(first_path), '--max-steps', '2']
    with pytest.raises(ValueError, match='--goal-detach-views differs from checkpoint'):
        train(parser().parse_args(resume + ['--goal-detach-views', WRISTS[0]]))
    assert load_checkpoint(train(parser().parse_args(resume)))['model_config']['goal_detach_views'] == WRISTS
    for name, extra, message in (('late', ['--goal-fusion', 'late', '--goal-stem-init', 'zero'], 'goal-fusion early'),
                                 ('depth', ['--goal-fusion-depth', '1'], 'goal-fusion-depth 0'),
                                 ('batch', ['--backbone-camera-batch'], 'one backbone pass per camera'),
                                 ('views', ['--goal-views', 'zed_link'], 'distinct views'),
                                 ('compile', ['--compile', 'default'], 'regions-autotune')):
        with pytest.raises(ValueError, match=message):
            train(parser().parse_args(common(tiny_root, tmp_path / name) + extra + ['--goal-detach-views', *WRISTS,
                                                                                  '--max-steps', '1']))
    plain = load_checkpoint(train(parser().parse_args(common(tiny_root, tmp_path / 'plain') + ['--max-steps', '1'])))
    assert 'goal_detach_views' not in plain['model_config'] and 'detach_views' not in plain['conditioning']['goal']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_recipe_paths_honour_the_option(tiny_root, tmp_path):
    """Fast stem, regions-autotune compile and bf16-backbone autocast: the compiled gradient is the eager detached one,
    not the plain one; then a 3-step training smoke with the recipe's flags."""
    x = batch('cuda')
    name = 'model.backbones.0.0.body.conv1.goal_weight'
    for autocast in (None, torch.bfloat16):
        options = {'backbone_autocast_dtype': autocast, 'fast_stem': True}
        eager, compiled = {}, {}
        for detach in (False, True):
            eager[detach] = gradients(policy(detach, 'cuda', **options), x)[name]
            model = policy(detach, 'cuda', **options)
            assert isinstance(model.model.backbones[0][0].body.conv1, PairedConv2d)  # the fast stem never replaces it
            compile_policy(model, 'regions-autotune')
            compiled[detach] = gradients(model, x)[name]
        noise = (compiled[False] - eager[False]).norm()  # compile's own numerical difference, without the option
        error, separation = (compiled[True] - eager[True]).norm(), (eager[False] - eager[True]).norm()
        assert error < 3 * noise + 1e-6 and error < 0.25 * separation, (autocast, float(error), float(noise), float(separation))
    output = tmp_path / 'cuda'
    path = train(parser().parse_args(common(tiny_root, output, device='cuda')
                                     + ['--goal-detach-views', *WRISTS, '--fast-stem', '--compile', 'regions-autotune',
                                        '--autocast', 'bf16-backbone', '--max-steps', '3']))
    checkpoint = load_checkpoint(path)
    assert checkpoint['step'] == 3 and checkpoint['model_config']['goal_detach_views'] == WRISTS
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values() if v.is_floating_point())
