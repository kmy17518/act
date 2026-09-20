"""CPU-only controlled initialization, held-out data, and diagnostic lifecycle tests."""

from collections import Counter, deque
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import b1k_compare_language_init as comparison
from b1k_dataset import B1KDataset, StepBatchSampler
from b1k_language import CLIP_MODEL, CLIP_REVISION, LONG_PROMPT_POLICY
from b1k_training import load_checkpoint, make_policy
from test_b1k import tiny_root


@pytest.fixture
def small_config():
    torch.set_num_threads(1)
    return dict(comparison.model_config(27), num_queries=4, hidden_dim=32, dim_feedforward=64,
                enc_layers=1, dec_layers=1, nheads=4, dropout=0.2, pretrained_backbone=False)


@pytest.fixture
def fake_batch():
    rng = torch.Generator().manual_seed(97)
    images = torch.rand(2, 3, 3, 32, 32, generator=rng)
    qpos = torch.randn(2, 27, generator=rng)
    qpos[:, -2:] = torch.eye(2)
    actions = torch.randn(2, 4, 23, generator=rng)
    padding = torch.tensor([[False, False, False, True], [False, False, True, True]])
    embeddings = torch.randn(2, 768, generator=rng)
    return (images, qpos, actions, padding), embeddings


def cache_value(task_map):
    return {'version': 1, 'model': CLIP_MODEL, 'revision': CLIP_REVISION, 'embedding_dim': 768,
            'normalized': False, 'long_prompt_policy': LONG_PROMPT_POLICY, 'prompt_source': 'task_name',
            'tasks': {key: {'task_name': name, 'prompt': name,
                            'embedding': (torch.arange(768).float() / 768 + key / 100).tolist()}
                      for key, name in task_map.items()}}


def test_full_architecture_and_short_defaults():
    args = comparison.parser().parse_args(['--output-dir', '/tmp/test-comparison'])
    config = comparison.model_config(26)
    assert (args.max_steps, args.batch_size, args.eval_samples, args.eval_every) == (1000, 64, 128, 250)
    assert args.holdout_every == 10 and args.wandb_mode == 'disabled'
    assert {key: config[key] for key in ('hidden_dim', 'dim_feedforward', 'enc_layers', 'dec_layers',
                                        'nheads', 'num_queries', 'kl_weight', 'lr', 'lr_backbone', 'weight_decay')} == {
        'hidden_dim': 512, 'dim_feedforward': 3200, 'enc_layers': 4, 'dec_layers': 7,
        'nheads': 8, 'num_queries': 100, 'kl_weight': 10, 'lr': 1e-5, 'lr_backbone': 1e-5,
        'weight_decay': 1e-4}
    assert comparison.IMAGE_SIZE == (240, 240)


def test_every_shared_parameter_buffer_byte_exact_and_only_film_differs(small_config):
    policies, proof = comparison.build_arms(small_config, 'cpu')
    assert proof['byte_exact'] and len(set(proof['shared_hashes'].values())) == 1
    assert proof['shared_buffer_tensors'] > 0
    assert 'model.pos_table' in proof['shared_state_keys']
    assert proof['identity_film_zero']
    baseline = policies['baseline'].state_dict()
    random_state = policies['random_film'].state_dict()
    identity = policies['identity_film'].state_dict()
    changed = set()
    for name in random_state:
        if comparison.tensor_bytes(random_state[name]) != comparison.tensor_bytes(identity[name]):
            changed.add(name)
        if name in baseline:
            assert comparison.tensor_bytes(random_state[name]) == comparison.tensor_bytes(baseline[name])
    assert changed == comparison.film_names(policies['random_film'])
    assert all(not identity[name].count_nonzero() for name in changed)
    for arm in comparison.ARMS:
        assert proof['optimizer'][arm]['every_trainable_parameter_once']
        assert proof['optimizer'][arm]['independent']
    buffer = next(iter(dict(policies['identity_film'].named_buffers()).values()))
    buffer.reshape(-1)[0] += 1
    with pytest.raises(AssertionError, match='shared state is not byte-exact'):
        comparison.prove_shared_state(policies)


def test_step_zero_exact_inference_loss_and_no_state_or_rng_mutation(small_config, fake_batch):
    policies, _ = comparison.build_arms(small_config, 'cpu')
    batch, embeddings = fake_batch
    before = {arm: comparison.state_hash(policy.state_dict()) for arm, policy in policies.items()}
    torch.manual_seed(772)
    rng = torch.get_rng_state().clone()
    proof = comparison.check_step_zero(policies, batch, embeddings, 0, torch.device('cpu'), rtol=0, atol=0)
    assert proof['passed'] and proof['same_forward_rng_consumption']
    assert proof['identity_inference_max_abs_error'] == 0
    assert set(proof['identity_train_loss_abs_errors'].values()) == {0}
    assert proof['random_inference_max_abs_difference'] > 0
    assert before == {arm: comparison.state_hash(policy.state_dict()) for arm, policy in policies.items()}
    assert torch.equal(rng, torch.get_rng_state())
    assert all(policy.training for policy in policies.values())
    assert all(not policy.configure_optimizers().state for policy in policies.values())
    assert not embeddings.requires_grad


def test_paired_stochastic_draws_shared_gradients_and_film_optimizer_update(small_config, fake_batch):
    policies, _ = comparison.build_arms(small_config, 'cpu')
    batch, embeddings = fake_batch
    images, qpos, actions, padding = batch
    device = torch.device('cpu')
    _, snapshot = comparison.step_rng(0, 1, device)
    posterior, dropout = {}, {}
    handles = []
    for arm, policy in policies.items():
        handles.append(policy.model.latent_out_proj.register_forward_pre_hook(
            lambda module, inputs, arm=arm: posterior.setdefault(arm, []).append(inputs[0].detach().clone())))
        layer = next(module for module in policy.model.encoder.modules() if isinstance(module, torch.nn.Dropout))
        handles.append(layer.register_forward_hook(
            lambda module, inputs, output, arm=arm: dropout.setdefault(arm, []).append(output.detach().clone())))
    try:
        for arm in ('baseline', 'identity_film'):
            comparison.restore_rng(snapshot, device)
            policy = policies[arm]
            losses = policy(qpos, images, actions, padding, **comparison.policy_kwargs(arm, qpos, embeddings))
            losses['loss'].backward()
        other = dict(policies['identity_film'].named_parameters())
        for name, value in policies['baseline'].named_parameters():
            if value.grad is not None:
                torch.testing.assert_close(value.grad, other[name].grad, rtol=2e-5, atol=2e-6)
        for policy in policies.values():
            policy.configure_optimizers().zero_grad(set_to_none=True)
        posterior.clear()
        dropout.clear()
        before = {arm: {name: value.clone() for name, value in policy.state_dict().items()
                        if name in comparison.film_names(policy)} for arm, policy in policies.items()}
        records = comparison.train_step(policies, batch, embeddings, 0, 1, device)
        assert records['baseline']['total'] == records['identity_film']['total']
        for arm in comparison.ARMS:
            torch.testing.assert_close(posterior['baseline'][0], posterior[arm][0], atol=0, rtol=0)
            torch.testing.assert_close(dropout['baseline'][0], dropout[arm][0], atol=0, rtol=0)
            assert records[arm]['grad_norm'] > 0
            assert records[arm]['total'] == pytest.approx(records[arm]['l1'] + 10 * records[arm]['kl'])
            assert records[arm]['loss'] == records[arm]['total']
            for name, value in policies[arm].named_parameters():
                if name in comparison.film_names(policies[arm]):
                    assert not torch.equal(before[arm][name], value)
                    state = policies[arm].configure_optimizers().state[value]
                    assert int(state['step']) == 1 and state['exp_avg_sq'].count_nonzero() > 0
            if arm != 'baseline':
                assert records[arm]['film_grad_norm'] > 0
        previous = posterior['baseline'][0].clone()
        _, different = comparison.step_rng(0, 2, device)
        comparison.restore_rng(different, device)
        policies['baseline'](qpos, images, actions, padding)
        assert not torch.equal(previous, posterior['baseline'][-1])
    finally:
        for handle in handles:
            handle.remove()


def test_evaluation_zero_latent_padding_weighting_and_rng_restoration(small_config, fake_batch):
    policies, _ = comparison.build_arms(small_config, 'cpu')
    batch, embeddings = fake_batch
    latents = []
    handles = [policy.model.latent_out_proj.register_forward_pre_hook(
        lambda module, inputs: latents.append(inputs[0].detach().clone())) for policy in policies.values()]
    torch.manual_seed(888)
    rng = torch.get_rng_state().clone()
    try:
        result = comparison.evaluate(policies, [batch], embeddings, torch.device('cpu'), 0)
        assert all(not latent.count_nonzero() for latent in latents)
        assert torch.equal(rng, torch.get_rng_state())
        assert all(policy.training for policy in policies.values())
        assert result['baseline']['inference_l1'] == result['identity_film']['inference_l1']
        assert result['baseline']['valid_action_values'] == 5 * 23
        assert result['baseline']['inference_l1_act_denominator'] == pytest.approx(
            result['baseline']['inference_l1'] * 5 / 8)
        second = comparison.evaluate(policies, [batch], embeddings, torch.device('cpu'), 9)
        assert result['baseline']['l1'] == second['baseline']['l1']
    finally:
        for handle in handles:
            handle.remove()


def test_split_train_only_stats_sampler_and_cache_reuse(tiny_root, tmp_path):
    dataset = B1KDataset(tiny_root, chunk_size=4, image_size=(32, 32))
    full_stats = dataset.compute_stats()
    dataset.stats = full_stats
    dataset[0]
    assert dataset._videos and dataset._groups
    evaluation, split = comparison.split_dataset(dataset, every=2)
    assert split['train_episode_ids'] == [42] and split['eval_episode_ids'] == [99]
    assert dataset._videos is not evaluation._videos and dataset._groups is not evaluation._groups
    assert dataset.stats is None and evaluation.stats is None
    assert not dataset._videos and not evaluation._videos
    assert list(dataset.by_id) == [42] and list(evaluation.by_id) == [99]
    assert dataset.lengths.tolist() == [5] and dataset.starts.tolist() == [0] and dataset.ends.tolist() == [5]
    path = tmp_path / 'train-stats.json'
    with patch.object(dataset, 'compute_stats', wraps=dataset.compute_stats) as compute:
        stats = comparison.training_stats(dataset, path)
        assert comparison.training_stats(dataset, path) == stats
    assert compute.call_count == 1 and stats['count'] == 5 and not stats['approximate']
    assert stats['fingerprint'] != full_stats['fingerprint']
    assert stats['action_mean'] != full_stats['action_mean']
    for indices in StepBatchSampler(dataset, 20, 0, 4, 0):
        assert all(0 <= index < 5 for index in indices)
        assert all(dataset.episodes[np.searchsorted(dataset.ends, index, side='right')]['episode_index'] == 42
                   for index in indices)
    path.write_text(json.dumps(full_stats))
    with pytest.raises(ValueError, match='training-only'):
        comparison.training_stats(dataset, path)
    dataset.close()
    evaluation.close()


def test_200_episode_split_and_fixed_128_eval_distribution():
    dataset = object.__new__(B1KDataset)
    dataset.episodes = [{'episode_index': i * 3, 'length': 100 + i} for i in range(200)]
    dataset.by_id = {ep['episode_index']: ep for ep in dataset.episodes}
    dataset._videos, dataset._groups, dataset._footers = {}, {}, {}
    dataset.frame_cache = dataset._table = dataset._frame_rows = None
    evaluation, split = comparison.split_dataset(dataset)
    assert evaluation.positions == {ep['episode_index']: i for i, ep in enumerate(evaluation.episodes)}
    assert dataset.positions == {ep['episode_index']: i for i, ep in enumerate(dataset.episodes)}
    assert split['eval_episode_ids'] == [i * 3 for i in range(9, 200, 10)]
    assert len(split['train_episode_ids']) == 180 and len(split['eval_episode_ids']) == 20
    assert not set(split['train_episode_ids']).intersection(split['eval_episode_ids'])
    samples = comparison.fixed_eval_samples(evaluation, 128, 10000)
    assert samples == comparison.fixed_eval_samples(evaluation, 128, 10000)
    assert samples != comparison.fixed_eval_samples(evaluation, 128, 10001)
    assert set(Counter(row['episode_id'] for row in samples).values()) == {6, 7}
    assert len({(row['episode_id'], row['frame']) for row in samples}) == 128


def test_cached_clip_never_constructs_encoder(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', None)
    task_map = {0: 'turning_on_radio'}
    path = tmp_path / 'source.json'
    path.write_text(json.dumps({'task_map': task_map, 'language_cache': cache_value(task_map)}))
    source, cache, embeddings, digest = comparison.cached_language(path, task_map)
    assert embeddings.shape == (1, 768) and not embeddings.requires_grad
    assert len(digest) == 64 and cache['tasks'][0]['prompt'] == 'turning_on_radio'
    with pytest.raises(ValueError, match='task map differs'):
        comparison.cached_language(path, {1: 'other'})


def test_final100_window():
    history = {arm: deque(maxlen=100) for arm in comparison.ARMS}
    for step in range(1, 126):
        for records in history.values():
            records.append(dict.fromkeys(('l1', 'kl', 'total', 'grad_norm', 'film_grad_norm'), step))
    summary = comparison.summarize(history)
    for arm in comparison.ARMS:
        assert summary[arm]['final100_count'] == 100
        assert summary[arm]['final100_mean']['total'] == 75.5


def test_cpu_lifecycle_checkpoint_summary_and_same_batch_all_arms(tiny_root, tmp_path, small_config, monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', None)
    monkeypatch.setattr(comparison, 'IMAGE_SIZE', (32, 32))
    monkeypatch.setattr(comparison, 'model_config', lambda state_dim: dict(small_config, state_dim=state_dim))
    cache_path = tmp_path / 'source.json'
    task_map = {7: 'first', 9: 'second'}
    cache_path.write_text(json.dumps({'task_map': task_map, 'language_cache': cache_value(task_map),
                                     'train_config': {'batch_size': 1560}}))
    output = tmp_path / 'comparison'
    stats_path = tmp_path / 'exact-train.json'
    args = comparison.parser().parse_args([
        '--dataset-path', str(tiny_root), '--task-names', 'first', 'second', '--expected-episodes', '2',
        '--holdout-every', '2', '--cache-run', str(cache_path), '--output-dir', str(output),
        '--max-steps', '2', '--batch-size', '2', '--num-workers', '0', '--device', 'cpu',
        '--eval-samples', '3', '--eval-batch-size', '2', '--eval-every', '1', '--torch-threads', '1',
        '--stats-cache', str(stats_path)])
    seen, policies_used = {}, {}
    original = comparison.train_step

    def train_step(policies, batch, *args):
        for arm, policy in policies.items():
            policies_used[arm] = policy
        pointers = []
        handles = []
        for policy in policies.values():
            handles.append(policy.model.register_forward_pre_hook(
                lambda module, inputs: pointers.append((inputs[0].data_ptr(), inputs[3].data_ptr(), inputs[4].data_ptr()))))
        try:
            result = original(policies, batch, *args)
        finally:
            for handle in handles:
                handle.remove()
        seen[args[2]] = pointers
        return result

    with patch.object(comparison, 'train_step', side_effect=train_step), patch.object(
            B1KDataset, 'compute_stats', autospec=True, side_effect=B1KDataset.compute_stats) as stats:
        summary = comparison.compare(args)
    assert stats.call_count == 1
    assert len(seen) == 2 and all(len(pointers) == 3 and len(set(pointers)) == 1 for pointers in seen.values())
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest == json.loads((output / 'run.json').read_text())
    assert manifest['status'] == 'completed' and manifest['step_zero']['passed']
    assert manifest['split']['normalization_scope'] == 'train_episodes_only'
    assert manifest['normalization']['count'] == 5
    assert manifest['precision'] == 'float32' and manifest['gradient_accumulation_steps'] == 1
    assert not manifest['autocast'] and not manifest['clip_encoder_loaded']
    assert manifest['long_run_batch_size'] == 1560 and manifest['diagnostic_batch_size'] == 2
    assert all(row['episode_id'] == 99 for row in manifest['eval_samples'])
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert len(records) == 6 and {row['step'] for row in records} == {1, 2}
    evaluations = [json.loads(line) for line in (output / 'eval.jsonl').read_text().splitlines()]
    assert len(evaluations) == 9 and {row['step'] for row in evaluations} == {0, 1, 2}
    assert summary == json.loads((output / 'summary.json').read_text())
    for arm, path in summary['checkpoints'].items():
        saved = load_checkpoint(path)
        assert saved['step'] == 2 and saved['comparison_arm'] == arm
        assert ('language_cache' in saved) == (arm != 'baseline')
        assert saved['normalization']['count'] == 5
        assert len(saved['optimizer']['state']) > 0
        for name, value in policies_used[arm].state_dict().items():
            torch.testing.assert_close(saved['model'][name], value, atol=0, rtol=0)
        mean = np.mean([row['total'] for row in records if row['arm'] == arm])
        assert summary['arms'][arm]['final100_mean']['total'] == mean
        assert summary['arms'][arm]['final100_count'] == 2
    with pytest.raises(FileExistsError, match='new empty'):
        comparison.compare(args)


def test_concurrent_wandb_handles_names_group_and_no_resume(tmp_path, monkeypatch):
    created = []

    class FakeRun:
        def __init__(self, kwargs):
            self.id = kwargs['id']
            self.entity = 'test'
            self.settings = SimpleNamespace(mode=kwargs['mode'])
            self.finished = False

        def define_metric(self, *args, **kwargs):
            pass

        def finish(self, **kwargs):
            self.finished = True

    def init(**kwargs):
        assert kwargs['reinit'] == 'create_new' and kwargs['resume'] == 'never'
        assert not any(run.finished for _, run in created)
        run = FakeRun(kwargs)
        created.append((kwargs, run))
        return run

    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init, Settings=lambda **kw: kw))
    args = comparison.parser().parse_args(['--output-dir', str(tmp_path), '--wandb-mode', 'offline',
                                           '--wandb-group', 'paired-init'])
    with comparison.comparison_wandb(args, tmp_path, {}, {'wandb': {'project': 'same-project'}}) as (runs, ids):
        assert set(runs) == set(comparison.ARMS)
        assert len({entry['id'] for entry in ids.values()}) == 3
        assert all(entry['group'] == 'paired-init' and entry['project'] == 'same-project' for entry in ids.values())
        assert all(entry['name'].startswith('act-init-20260917-') for entry in ids.values())
    assert all(run.finished for _, run in created)


def test_default_cuda_resolves_index_without_gpu():
    with patch.object(torch.cuda, 'current_device', return_value=2) as current:
        assert comparison.resolve_device('cuda') == torch.device('cuda:2')
        current.assert_called_once()
    with patch.object(torch.cuda, 'current_device', side_effect=AssertionError('must not query CUDA')):
        assert comparison.resolve_device('cuda:0') == torch.device('cuda:0')
        assert comparison.resolve_device('cpu') == torch.device('cpu')
    with pytest.raises(ValueError, match='Only CPU and CUDA'):
        comparison.resolve_device('mps')


@pytest.mark.parametrize('option,value', [('--max-steps', '0'), ('--eval-every', '0'), ('--batch-size', '0'),
                                         ('--seed', '-1'), ('--num-workers', '-1'), ('--parity-atol', '1e-3')])
def test_invalid_arguments_fail_before_model_or_data(tmp_path, option, value):
    args = comparison.parser().parse_args(['--output-dir', str(tmp_path / 'new'), option, value])
    with patch.object(comparison, 'B1KDataset', side_effect=AssertionError('must fail before dataset')):
        with pytest.raises(ValueError):
            comparison.compare(args)
