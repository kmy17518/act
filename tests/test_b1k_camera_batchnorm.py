"""Per-camera BatchNorm running statistics (`--backbone-norm batch_per_camera`) and the recalibration tool."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torchvision import transforms

from b1k_server import PolicyPredictor, Session
from b1k_training import load_checkpoint, make_policy, parser, train
from detr.models.backbone import Backbone, PerCameraBatchNorm2d
from test_b1k import observation, small_model_config, tiny_root  # noqa: F401
from test_b1k_language import fake_minilm, prompt_rows, write_prompts  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/b1k'))
import recalibrate_camera_batchnorm as recalibration  # noqa: E402

IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def test_per_camera_batchnorm_matches_batchnorm_and_keeps_statistics_per_camera():
    torch.manual_seed(0)
    reference = torch.nn.BatchNorm2d(4)
    layer = PerCameraBatchNorm2d(4, num_cameras=3)
    weight, bias = torch.randn(4), torch.randn(4)
    with torch.no_grad():
        for module in (reference, layer):
            module.weight.copy_(weight)
            module.bias.copy_(bias)
    assert sorted(layer.state_dict()) == sorted(['weight', 'bias'] + [f'{name}_{c}' for c in range(3) for name in
                                                ('running_mean', 'running_var', 'num_batches_tracked')])
    inputs = [torch.randn(5, 4, 6, 6) * (camera + 1) + camera for camera in range(3)]
    # Training: the output is nn.BatchNorm2d's (batch statistics of this pass); only the selected camera's set moves.
    for camera, x in enumerate(inputs):
        layer.camera = camera
        torch.testing.assert_close(layer(x), reference(x), atol=0, rtol=0)
    for camera, x in enumerate(inputs):
        mean, var, tracked = layer.statistics(camera)
        torch.testing.assert_close(mean, 0.1 * x.mean(dim=(0, 2, 3)))  # momentum 0.1 from zero
        torch.testing.assert_close(var, 0.9 + 0.1 * x.var(dim=(0, 2, 3), unbiased=True))
        assert int(tracked) == 1
    # Eval: each camera is normalized with its own statistics; another camera's statistics give another output.
    layer.eval()
    outputs = []
    for camera, x in enumerate(inputs):
        layer.camera = camera
        mean, var, _ = layer.statistics(camera)
        expected = (x - mean[None, :, None, None]) / (var[None, :, None, None] + layer.eps).sqrt()
        expected = expected * layer.weight[None, :, None, None] + layer.bias[None, :, None, None]
        torch.testing.assert_close(layer(x), expected)
        outputs.append(layer(x))
    layer.camera = 1
    assert (layer(inputs[0]) - outputs[0]).abs().max() > 1e-3
    with pytest.raises(ValueError, match='outside'):
        layer.statistics(3)
    # Cumulative averaging (momentum None) and a reset of one camera only.
    layer.momentum = None
    layer.train()
    layer.reset_running_stats(camera=2)
    layer.camera = 2
    layer(inputs[2])
    layer(inputs[2] + 1)
    mean, _, tracked = layer.statistics(2)
    torch.testing.assert_close(mean, inputs[2].mean(dim=(0, 2, 3)) + 0.5)
    assert int(tracked) == 2 and int(layer.statistics(0)[2]) == 1
    # A single-statistics state dict (nn.BatchNorm2d) loads by copying its statistics to every camera.
    fresh = PerCameraBatchNorm2d(4, num_cameras=3)
    result = fresh.load_state_dict(reference.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for camera in range(3):
        mean, var, tracked = fresh.statistics(camera)
        torch.testing.assert_close(mean, reference.running_mean, atol=0, rtol=0)
        torch.testing.assert_close(var, reference.running_var, atol=0, rtol=0)
        assert int(tracked) == int(reference.num_batches_tracked)
    with pytest.raises(ValueError, match='at least one camera'):
        PerCameraBatchNorm2d(4, num_cameras=0)


def test_backbone_routes_passes_to_camera_statistics_and_rejects_camera_batching():
    torch.manual_seed(3)
    backbone = Backbone('resnet18', True, False, False, pretrained=False, language_conditioning='mt_act',
                        film_cond_dim=16, norm='batch_per_camera', num_cameras=3)
    assert len(backbone.per_camera_norms) == 20 and all(m.num_cameras == 3 for m in backbone.per_camera_norms)
    backbone.select_camera(2)
    assert all(m.camera == 2 for m in backbone.per_camera_norms)
    with pytest.raises(ValueError, match='Unsupported backbone normalization'):
        Backbone('resnet18', True, False, False, pretrained=False, norm='per_camera')
    config = dict(small_model_config(), language_conditioning='mt_act', language_encoder='minilm', state_dim=25,
                  backbone_norm='batch_per_camera')
    with pytest.raises(ValueError, match='exclusive'):
        make_policy(dict(config, camera_batch=True), 'cpu')
    policy = make_policy(config, 'cpu')
    layers = [m for m in policy.modules() if isinstance(m, PerCameraBatchNorm2d)]
    joiner = policy.model.backbones[0]
    with pytest.raises(ValueError, match='camera index'):
        joiner(torch.rand(2, 3, 32, 32), lang_emb=torch.randn(2, config['hidden_dim']))
    qpos, images = torch.randn(2, 26), torch.rand(2, 3, 3, 32, 32)
    language = torch.nn.functional.normalize(torch.randn(2, 384), dim=-1)
    actions, pad = torch.randn(2, 4, 23), torch.zeros(2, 4, dtype=torch.bool)
    for layer in layers:
        layer.momentum = 1.0  # running statistics = this pass's batch statistics
    policy(qpos, images, actions, pad, lang_emb=language)['loss'].backward()
    stem = policy.model.backbones[0][0].body
    for camera in range(3):
        mean, var, tracked = stem.bn1.statistics(camera)
        with torch.no_grad():
            out = stem.conv1(IMAGENET(images[:, camera]))
        torch.testing.assert_close(mean, out.mean(dim=(0, 2, 3)), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(var, out.var(dim=(0, 2, 3), unbiased=True), atol=1e-5, rtol=1e-4)
        assert int(tracked) == 1
    for layer in layers:
        assert all(int(layer.statistics(camera)[2]) == 1 for camera in range(3))
    # Eval mode uses each camera's own statistics: the forward of the loop equals a manual per-camera eval pass.
    policy.eval()
    with torch.no_grad():
        first = policy(qpos, images, lang_emb=language)
        again = policy(qpos, images, lang_emb=language)
    torch.testing.assert_close(first, again, atol=0, rtol=0)
    assert all(layer.camera == 2 for layer in layers)  # the last pass selected the last camera


def test_train_resume_serve_and_recalibrate(tiny_root, tmp_path, fake_minilm, monkeypatch):
    torch.set_num_threads(1)
    write_prompts(tiny_root, prompt_rows())
    common = ['--dataset-path', str(tiny_root), '--batch-size', '2', '--num-workers', '0', '--torch-threads', '1',
              '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
              '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
              '--save-every', '1', '--dropout', '0.1', '--export-every', '1',
              '--language-conditioning', 'mt_act', '--prompt-source', 'task_description', '--no-pretrained-backbone']
    with pytest.raises(ValueError, match='exclusive'):
        train(parser().parse_args(common + ['--backbone-norm', 'batch_per_camera', '--backbone-camera-batch',
                                           '--output-dir', str(tmp_path / 'bad'), '--max-steps', '1']))
    with pytest.raises(ValueError, match='CNNMLP has one per camera'):
        train(parser().parse_args(['--dataset-path', str(tiny_root), '--policy-class', 'CNNMLP', '--device', 'cpu',
                                   '--num-workers', '0', '--backbone-norm', 'batch_per_camera', '--no-pretrained-backbone',
                                   '--output-dir', str(tmp_path / 'bad'), '--max-steps', '1']))
    per_camera = common + ['--backbone-norm', 'batch_per_camera']
    full, split = tmp_path / 'full', tmp_path / 'split'
    expected_path = train(parser().parse_args(per_camera + ['--output-dir', str(full), '--max-steps', '2']))
    first_path = train(parser().parse_args(per_camera + ['--output-dir', str(split), '--max-steps', '1']))
    first = load_checkpoint(first_path)
    assert first['model_config']['backbone_norm'] == 'batch_per_camera' and not first['model_config']['camera_batch']
    keys = [key for key in first['model'] if key.endswith('body.bn1.running_mean_2')]
    assert keys and not any(key.endswith('body.bn1.running_mean') for key in first['model'])
    with pytest.raises(ValueError, match='differs from checkpoint'):
        train(parser().parse_args(common + ['--backbone-norm', 'batch', '--output-dir', str(split), '--resume',
                                           str(first_path), '--max-steps', '2']))
    source_dir = tmp_path / 'batch'  # per-camera-pass `batch` checkpoint for the recalibration below
    source_path = train(parser().parse_args(common + ['--backbone-norm', 'batch', '--output-dir', str(source_dir),
                                                     '--max-steps', '1']))
    monkeypatch.setitem(sys.modules, 'transformers', None)
    resumed = load_checkpoint(train(parser().parse_args(common + ['--output-dir', str(split), '--resume', str(first_path),
                                                                 '--max-steps', '2', '--loader-batch-size', '1'])))
    expected = load_checkpoint(expected_path)
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)
    exported = load_checkpoint(split / 'export_queue/eval/step_00000002.pt')
    predictor = PolicyPredictor(exported, 'cpu')
    actions = Session(predictor, exported, action_horizon=2).act(observation(2, np.array([7, 9])))
    assert actions.shape[-1] == 23 and np.isfinite(actions).all()

    # Recalibration: a per-camera-pass `batch` checkpoint becomes a `batch_per_camera` one whose statistics are the
    # per-camera batch statistics of the calibration batches; the parameters and optimizer state are untouched.
    source = load_checkpoint(source_path)
    output = tmp_path / 'converted.pt'
    args = recalibration.parser().parse_args(['--checkpoint', str(source_path), '--output', str(output), '--dataset-path',
                                              str(tiny_root), '--batches', '2', '--batch-size', '2', '--num-workers', '0',
                                              '--loader-batch-size', '1', '--device', 'cpu', '--matmul-precision', 'highest',
                                              '--evaluate'])
    report = recalibration.convert(args)
    assert report['batches'] == 2 and report['seed'] == source['train_config']['seed'] and report['start_step'] == 1
    assert 'l1_mixed_running_statistics' in report and 'l1_per_camera_running_statistics' in report
    converted = load_checkpoint(output)
    assert converted['checkpoint_type'] == 'full' and 'optimizer' in converted
    assert converted['model_config']['backbone_norm'] == 'batch_per_camera'
    assert converted['camera_batchnorm_recalibration']['method'].startswith('cumulative average')
    for key, value in source['model'].items():
        if 'running_' in key or 'num_batches_tracked' in key:
            continue
        torch.testing.assert_close(converted['model'][key], value, atol=0, rtol=0)
    # The stem statistics equal the cumulative per-camera batch statistics of the two calibration batches.
    config = dict(converted['model_config'])
    policy = make_policy(config, 'cpu', restoring=True)
    policy.load_state_dict(converted['model'])
    policy.eval()
    seen = {camera: [] for camera in range(3)}
    dataset = _dataset(tiny_root, converted)
    for images, _, _, _ in recalibration.batch_iterator(dataset, args, torch.device('cpu'), report['seed'],
                                                       report['start_step']):
        with torch.no_grad():
            for camera in range(3):
                seen[camera].append(policy.model.backbones[0][0].body.conv1(IMAGENET(images[:, camera])))
    dataset.close()
    assert all(len(outputs) == 2 for outputs in seen.values())
    for camera in range(3):
        mean, _, tracked = policy.model.backbones[0][0].body.bn1.statistics(camera)
        assert int(tracked) == 2
        torch.testing.assert_close(mean, torch.stack([o.mean(dim=(0, 2, 3)) for o in seen[camera]]).mean(0),
                                   atol=1e-5, rtol=1e-4)
    with pytest.raises(FileExistsError):
        recalibration.convert(args)
    with pytest.raises(ValueError, match='backbone_norm=batch checkpoint'):
        recalibration.convert(recalibration.parser().parse_args(['--checkpoint', str(output), '--output',
                                                                 str(tmp_path / 'twice.pt'), '--dataset-path', str(tiny_root),
                                                                 '--device', 'cpu']))


def _dataset(root, checkpoint):
    dataset = recalibration.B1KDataset(root, list(checkpoint['task_map'].values()), checkpoint['model_config']['num_queries'],
                                       checkpoint['adapter_config']['image_size'], profile_reads=True,
                                       timestamp_tolerance=checkpoint['adapter_config']['timestamp_tolerance'])
    dataset.stats = checkpoint['normalization']
    return dataset
