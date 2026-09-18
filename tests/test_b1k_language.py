"""Offline CLIP prompt/cache, FiLM, and standalone lifecycle regressions."""

import asyncio
import copy
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import websockets.asyncio.client
import websockets.asyncio.server

from b1k_language import (CLIP_MODEL, CLIP_REVISION, build_language_cache, encode_prompt,
                          language_embedding_table, language_for_qpos, task_prompts)
from b1k_server import B1KServer, PolicyPredictor, Session, packb, unpackb
from b1k_training import load_checkpoint, make_policy, parser, save_eval_checkpoint, train
from detr.models.backbone import Backbone, FiLMLayer
from test_b1k import observation, small_model_config, tiny_root


class FakeTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 99, 100, 100

    def __call__(self, text, add_special_tokens=True, return_tensors=None, **kwargs):
        assert not kwargs.get('truncation')
        ids = [sum(word.encode()) % 90 + 1 for word in text.split()]
        if not add_special_tokens:
            return {'input_ids': ids}
        ids = [self.bos_token_id, *ids, self.eos_token_id]
        return self.pad({'input_ids': [ids], 'attention_mask': [[1] * len(ids)]}, return_tensors=return_tensors)

    def pad(self, tokens, padding=True, return_tensors='pt'):
        length = max(map(len, tokens['input_ids']))
        return {'input_ids': torch.tensor([ids + [self.pad_token_id] * (length - len(ids))
                                          for ids in tokens['input_ids']]),
                'attention_mask': torch.tensor([mask + [0] * (length - len(mask))
                                               for mask in tokens['attention_mask']])}


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(1, 769, dtype=torch.float32) / 768)
        self.calls = []

    def forward(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone(), self.training,
                           torch.is_grad_enabled(), self.weight.requires_grad))
        tokens = input_ids * attention_mask
        positions = torch.arange(1, input_ids.shape[1] + 1)
        value = (tokens * positions).sum(-1).float() / 1000
        return SimpleNamespace(text_embeds=torch.sin(value[:, None] + self.weight[None]))


@pytest.fixture
def fake_clip(monkeypatch):
    tokenizer, model = FakeTokenizer(), FakeEncoder()
    loaded = []

    def load_tokenizer(name, revision):
        loaded.append(('tokenizer', name, revision))
        return tokenizer

    def load_model(name, revision):
        loaded.append(('model', name, revision))
        return model

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
        CLIPTextModelWithProjection=SimpleNamespace(from_pretrained=load_model)))
    return tokenizer, model, loaded


def write_prompts(root, rows):
    (root / 'meta').mkdir(exist_ok=True)
    (root / 'meta/tasks.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def prompt_rows():
    return [{'task_index': 7, 'task_name': 'first', 'task': '  Move the first object.  '},
            {'task_index': 9, 'task_name': 'second', 'task': ' '.join(f'word{i}' for i in range(159))},
            {'task_index': 11, 'task_name': 'unavailable', 'task': ''}]


def test_prompt_sources_raw_names_exact_descriptions_and_selected_subset(tmp_path):
    task_map = {9: 'second', 7: 'first'}
    assert task_prompts(tmp_path, {7: 'raw_snake_case'}, 'task_name') == {7: 'raw_snake_case'}
    with pytest.raises(ValueError, match='descriptions require'):
        task_prompts(tmp_path, task_map, 'task_description')
    rows = prompt_rows()
    write_prompts(tmp_path, rows)
    assert task_prompts(tmp_path, task_map, 'task_description') == {7: rows[0]['task'], 9: rows[1]['task']}
    assert task_prompts(tmp_path, {7: 'first'}, 'task_description') == {7: rows[0]['task']}
    rows[1]['task'] = rows[0]['task']
    write_prompts(tmp_path, rows)
    assert len(task_prompts(tmp_path, task_map, 'task_description')) == 2
    (tmp_path / 'meta/tasks.jsonl').write_text('malformed unrelated sidecar')
    assert task_prompts(tmp_path, task_map, 'task_name') == task_map


@pytest.mark.parametrize('rows,error', [
    ([{'task_index': 7, 'task_name': 'first', 'task': 'Do it.'}] * 2, 'duplicate'),
    ([{'task_index': 7, 'task_name': 'wrong', 'task': 'Do it.'}], 'conflicting'),
    ([{'task_index': 70, 'task_name': 'first', 'task': 'Do it.'}], 'conflicting'),
    ([{'task_index': '7', 'task_name': 'first', 'task': 'Do it.'}], 'conflicting'),
    ([{'task_index': [7], 'task_name': 'first', 'task': 'Do it.'}], 'invalid'),
    ([{'task_index': 7, 'task_name': ['first'], 'task': 'Do it.'}], 'invalid'),
    ([{'task_index': 7, 'task_name': 'first', 'task': ' '}], 'missing'),
    ([{'task_index': 7, 'task_name': 'first'}], 'missing'),
    ([], 'missing'),
])
def test_selected_description_errors(tmp_path, rows, error):
    write_prompts(tmp_path, rows)
    with pytest.raises(ValueError, match=error):
        task_prompts(tmp_path, {7: 'first'}, 'task_description')


def test_clip_frozen_once_per_task_short_exact_and_long_mean(tmp_path, fake_clip):
    tokenizer, model, loaded = fake_clip
    rows = prompt_rows()
    write_prompts(tmp_path, rows)
    cache = build_language_cache(tmp_path, {7: 'first', 9: 'second'}, 'task_description')
    assert loaded == [('tokenizer', CLIP_MODEL, CLIP_REVISION), ('model', CLIP_MODEL, CLIP_REVISION)]
    assert len(model.calls) == 2
    assert all(not training and not grad and not requires for _, _, training, grad, requires in model.calls)
    ids, masks, *_ = model.calls[1]
    assert ids.shape == (3, 77) and masks.sum(-1).tolist() == [77, 77, 11]
    assert (ids[:, 0] == tokenizer.bos_token_id).all()
    assert all(ids[i, int(mask.sum()) - 1] == tokenizer.eos_token_id for i, mask in enumerate(masks))
    expected_short = model(**tokenizer(rows[0]['task'], return_tensors='pt')).text_embeds[0]
    torch.testing.assert_close(torch.tensor(cache['tasks'][7]['embedding']), expected_short, atol=0, rtol=0)
    words = rows[1]['task'].split()
    independent = [model(**tokenizer(' '.join(words[i:i + 75]))).text_embeds[0]
                   for i in range(0, len(words), 75)]
    torch.testing.assert_close(torch.tensor(cache['tasks'][9]['embedding']), torch.stack(independent).mean(0),
                               atol=0, rtol=0)
    table = language_embedding_table({'language_conditioning': 'clip_film', 'prompt_source': 'task_description'},
                                    {9: 'second', 7: 'first'}, cache)
    assert table.shape == (2, 768) and not torch.isclose(table.norm(dim=1), torch.ones(2)).any()
    qpos = torch.zeros(3, 27)
    qpos[:, -2:] = torch.tensor([[0., 1.], [1., 0.], [0., 1.]])
    torch.testing.assert_close(language_for_qpos(qpos, table), table[[1, 0, 1]])
    assert language_for_qpos(qpos, None) is None


def test_nonfinite_encoder_rejected(fake_clip):
    tokenizer, model, _ = fake_clip
    with torch.no_grad():
        model.weight[0] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        encode_prompt('test', tokenizer, model)


@pytest.mark.parametrize('damage', ['missing', 'source', 'model', 'revision', 'policy', 'task_ids', 'task_name',
                                    'prompt', 'shape', 'nan', 'inf'])
def test_invalid_language_cache_rejected(tmp_path, fake_clip, damage):
    task_map = {7: 'first'}
    cache = build_language_cache(tmp_path, task_map, 'task_name')
    config = {'language_conditioning': 'clip_film', 'prompt_source': 'task_name'}
    if damage == 'missing':
        cache = None
    elif damage in ('source', 'model', 'revision', 'policy'):
        key = {'source': 'prompt_source', 'policy': 'long_prompt_policy'}.get(damage, damage)
        cache[key] = 'different'
    elif damage == 'task_ids':
        cache['tasks'][8] = cache['tasks'].pop(7)
    elif damage in ('task_name', 'prompt'):
        cache['tasks'][7][damage] = 'different'
    elif damage == 'shape':
        cache['tasks'][7]['embedding'].pop()
    else:
        cache['tasks'][7]['embedding'][0] = float(damage)
    with pytest.raises(ValueError):
        language_embedding_table(config, task_map, cache)


def test_film_formula_and_checkpointed_gradient_parity():
    torch.set_num_threads(1)
    torch.manual_seed(5)
    film = FiLMLayer(3)
    x, language = torch.rand(2, 3, 4, 4), torch.randn(2, 768)
    beta, gamma = film.lang_proj(language).reshape(2, 6, 1, 1).chunk(2, 1)
    torch.testing.assert_close(film(x, language), torch.relu((1 + gamma) * x + beta), atol=0, rtol=0)
    checkpointed = Backbone('resnet18', True, False, False, pretrained=False, language_conditioning='clip_film')
    direct = copy.deepcopy(checkpointed)
    assert len([m for m in checkpointed.modules() if isinstance(m, FiLMLayer)]) == 8
    image = torch.rand(2, 3, 32, 32)
    first_language = language.clone().requires_grad_()
    second_language = language.clone().requires_grad_()
    actual = checkpointed(image, first_language)['0']
    actual.square().mean().backward()
    with patch('detr.models.backbone.checkpoint', side_effect=lambda fn, *a, **kw: fn(*a)) as mocked:
        expected = direct(image, second_language)['0']
        expected.square().mean().backward()
    assert mocked.call_count == 8
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(first_language.grad, second_language.grad, atol=0, rtol=0)
    for (name, p), (other_name, q) in zip(checkpointed.named_parameters(), direct.named_parameters()):
        assert name == other_name and p.grad is not None
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        if 'lang_proj' in name:
            assert p.grad.abs().sum() > 0


def test_film_identity_init_matches_unconditioned_and_recompute_is_runtime_only(tiny_root, tmp_path, fake_clip):
    torch.set_num_threads(1)
    with pytest.raises(ValueError, match='FiLM initialization'):
        FiLMLayer(3, init='other')
    film = FiLMLayer(3, init='identity')
    assert not film.lang_proj.weight.any() and not film.lang_proj.bias.any()
    x = torch.rand(2, 3, 4, 4)
    torch.testing.assert_close(film(x, torch.randn(2, 768)), x, atol=0, rtol=0)
    config = dict(small_model_config(), language_conditioning='clip_film')
    torch.manual_seed(21)
    baseline = make_policy(dict(config, language_conditioning='none'), 'cpu')
    torch.manual_seed(21)
    identity = make_policy(dict(config, film_init='identity'), 'cpu')
    torch.manual_seed(21)
    random_film = make_policy(dict(config, film_init='random'), 'cpu')
    assert all(not value.any() for key, value in identity.state_dict().items() if 'lang_proj' in key)
    assert any(value.any() for key, value in random_film.state_dict().items() if 'lang_proj' in key)
    mismatch = identity.load_state_dict(baseline.state_dict(), strict=False)
    assert not mismatch.unexpected_keys and all('lang_proj' in key for key in mismatch.missing_keys)
    qpos, images, language = torch.randn(2, 27), torch.rand(2, 3, 3, 32, 32), torch.randn(2, 768)
    baseline.eval()
    identity.eval()
    with torch.no_grad():
        torch.testing.assert_close(identity(qpos, images, lang_emb=language), baseline(qpos, images), atol=0, rtol=0)
    with pytest.raises(ValueError, match='Unsupported FiLM initialization'):
        make_policy(dict(config, film_init='other'), 'cpu')
    # Recomputation is a runtime attribute: same outputs and gradients, checkpoint() only when enabled.
    stored = make_policy(config, 'cpu', film_recompute=False)
    recomputed = copy.deepcopy(stored)
    for backbone in recomputed.model.backbones:
        backbone[0].body.recompute = True
    assert all(backbone[0].body.recompute is False for backbone in stored.model.backbones)
    actions, pad = torch.randn(2, 4, 23), torch.zeros(2, 4, dtype=torch.bool)
    torch.manual_seed(3)
    with patch('detr.models.backbone.checkpoint', side_effect=AssertionError('must not checkpoint')):
        stored(qpos, images, actions, pad, lang_emb=language)['loss'].backward()
    torch.manual_seed(3)
    recomputed(qpos, images, actions, pad, lang_emb=language)['loss'].backward()
    for (name, p), (other, q) in zip(stored.named_parameters(), recomputed.named_parameters()):
        assert name == other and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        assert p.grad is not None or 'is_pad_head' in name  # the unused pad head is the only gradient-free parameter
    # Trainer: the initialization is saved model configuration; recomputation is not.
    common = ['--dataset-path', str(tiny_root), '--batch-size', '2', '--num-workers', '0', '--torch-threads', '1',
              '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
              '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
              '--no-pretrained-backbone', '--save-every', '1', '--output-dir', str(tmp_path / 'run')]
    with pytest.raises(ValueError, match='requires --language-conditioning clip_film'):
        train(parser().parse_args(common + ['--max-steps', '1', '--film-init', 'identity']))
    path = train(parser().parse_args(common + ['--max-steps', '1', '--language-conditioning', 'clip_film',
                                              '--film-init', 'identity', '--no-film-recompute']))
    saved = load_checkpoint(path)
    assert saved['model_config']['film_init'] == 'identity'
    assert 'film_init' not in saved['train_config'] or saved['train_config']['film_init'] == 'identity'
    assert saved['train_config']['film_recompute'] is False and 'film_recompute' not in saved['model_config']
    with pytest.raises(ValueError, match='--film-init differs from checkpoint'):
        train(parser().parse_args(common + ['--max-steps', '2', '--resume', str(path), '--film-init', 'random']))
    unconditioned = train(parser().parse_args(common[:-2] + ['--output-dir', str(tmp_path / 'none'), '--max-steps', '1']))
    assert 'film_init' not in load_checkpoint(unconditioned)['model_config']


def test_conditioned_act_language_sensitivity_gradients_and_missing_input():
    torch.set_num_threads(1)
    torch.manual_seed(6)
    policy = make_policy(dict(small_model_config(), language_conditioning='clip_film'), 'cpu')
    qpos = torch.randn(2, 27)
    images = torch.rand(2, 3, 3, 32, 32)
    language = torch.randn(2, 768, requires_grad=True)
    actions = torch.randn(2, 4, 23)
    pad = torch.zeros(2, 4, dtype=torch.bool)
    policy(qpos, images, actions, pad, lang_emb=language)['loss'].backward()
    assert language.grad.abs().sum() > 0
    for name, value in policy.named_parameters():
        if 'lang_proj' in name:
            assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    policy.eval()
    with torch.no_grad():
        first = policy(qpos, images, lang_emb=language)
        different = policy(qpos, images, lang_emb=-language)
    assert (first - different).abs().max() > 1e-6
    with pytest.raises(ValueError, match='requires language embeddings'):
        policy(qpos, images)


@pytest.mark.parametrize('state_dim', [14, 27])
def test_none_legacy_keys_numerics_and_lazy_import(monkeypatch, state_dim):
    monkeypatch.setitem(sys.modules, 'transformers', None)
    config = dict(small_model_config(), state_dim=state_dim)
    torch.manual_seed(17)
    legacy = make_policy(config, 'cpu')
    torch.manual_seed(17)
    explicit = make_policy(dict(config, language_conditioning='none', prompt_source='task_description'), 'cpu')
    assert list(legacy.state_dict()) == list(explicit.state_dict())
    assert not any('film' in key for key in legacy.state_dict())
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    qpos, images = torch.rand(2, state_dim), torch.rand(2, 3, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(legacy(qpos, images), explicit(qpos, images), atol=0, rtol=0)
    with pytest.raises(ValueError, match='require clip_film'):
        explicit(qpos, images, lang_emb=torch.zeros(2, 768))
    with pytest.raises(ValueError, match='only supported for ACT'):
        make_policy(dict(config, policy_class='CNNMLP', num_queries=1, language_conditioning='clip_film'), 'cpu')
    assert parser().get_default('language_conditioning') == 'none'
    assert parser().get_default('prompt_source') == 'task_name'


@pytest.mark.parametrize('source', ['task_name', 'task_description'])
def test_conditioned_train_resume_eval_and_network_without_clip_or_sidecar(tiny_root, tmp_path, fake_clip,
                                                                          monkeypatch, source):
    torch.set_num_threads(1)
    write_prompts(tiny_root, prompt_rows())
    tokenizer, encoder, loaded = fake_clip
    common = ['--dataset-path', str(tiny_root), '--batch-size', '2', '--num-workers', '0', '--torch-threads', '1',
              '--device', 'cpu', '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32',
              '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1', '--nheads', '4',
              '--no-pretrained-backbone', '--save-every', '1', '--dropout', '0.1', '--export-every', '1']
    language_args = ['--language-conditioning', 'clip_film', '--prompt-source', source]
    full, split = tmp_path / 'full', tmp_path / 'split'
    expected_path = train(parser().parse_args(common + language_args +
                                            ['--output-dir', str(full), '--max-steps', '2']))
    first_path = train(parser().parse_args(common + language_args +
                                         ['--output-dir', str(split), '--max-steps', '1']))
    assert len(loaded) == 4 and len(encoder.calls) == 4
    first = load_checkpoint(first_path)
    assert first['model_config']['state_dim'] == 27
    assert first['adapter_config']['task_conditioning'] == 'onehot'
    assert first['language_cache']['prompt_source'] == source
    for option, value in [('--language-conditioning', 'none'),
                          ('--prompt-source', 'task_name' if source == 'task_description' else 'task_description')]:
        with pytest.raises(ValueError, match='differs from checkpoint'):
            train(parser().parse_args(common + ['--output-dir', str(split), '--resume', str(first_path),
                                               '--max-steps', '2', option, value]))
    if source == 'task_description':
        rows = prompt_rows()
        rows[0]['task'] = 'A changed instruction.'
        write_prompts(tiny_root, rows)
        with pytest.raises(ValueError, match='Resume language prompts differ'):
            train(parser().parse_args(common + ['--output-dir', str(split), '--resume', str(first_path),
                                               '--max-steps', '2']))
    (tiny_root / 'meta/tasks.jsonl').unlink()
    monkeypatch.setitem(sys.modules, 'transformers', None)
    with patch('b1k_language.load_clip_text_encoder', side_effect=AssertionError('CLIP must not load')):
        resumed_path = train(parser().parse_args(common + ['--output-dir', str(split), '--resume', str(first_path),
                                                          '--max-steps', '2', '--loader-batch-size', '1']))
        resumed = load_checkpoint(resumed_path)
        expected = load_checkpoint(expected_path)
        exported = load_checkpoint(split / 'export_queue/eval/step_00000002.pt')
        assert resumed['language_cache'] == expected['language_cache'] == exported['language_cache']
        assert resumed['model_config']['prompt_source'] == source
        for key in expected['model']:
            torch.testing.assert_close(expected['model'][key], resumed['model'][key], atol=0, rtol=0)
            torch.testing.assert_close(exported['model'][key], resumed['model'][key], atol=0, rtol=0)
        assert any('lang_proj' in key for key in exported['model'])
        assert not any('text_model' in key for key in exported['model'])
        assert 'optimizer' not in exported
        tiny_root.rename(tmp_path / 'hidden_dataset')
        predictor = PolicyPredictor(exported, 'cpu')
        restored = PolicyPredictor(resumed, 'cpu')
        obs = observation(2, np.array([7, 9]))
        reference = Session(restored, resumed, action_horizon=2).act(obs)
        np.testing.assert_array_equal(Session(predictor, exported, action_horizon=2).act(obs), reference)
        for temporal in (False, True):
            session = Session(predictor, exported, temporal_agg=temporal)
            assert np.isfinite(session.act(obs)).all()
            assert np.isfinite(session.act(observation(2, np.array([9, 9])))).all()
            session.reset()
            assert np.isfinite(session.act(observation())).all()
        metadata = {key: exported[key] for key in
                    ('model_config', 'adapter_config', 'task_map', 'normalization', 'language_cache')}
        save_eval_checkpoint(metadata, predictor.policy, 2, split)
        changed = copy.deepcopy(metadata)
        changed['language_cache']['tasks'][7]['embedding'][0] += 1
        with pytest.raises(FileExistsError, match='eval export differs'):
            save_eval_checkpoint(changed, predictor.policy, 2, split)

        async def network():
            server = B1KServer(exported, predictor, action_horizon=2)
            async with websockets.asyncio.server.serve(server.handler, '127.0.0.1', 0) as running:
                port = running.sockets[0].getsockname()[1]
                async with websockets.asyncio.client.connect(f'ws://127.0.0.1:{port}') as client:
                    metadata = unpackb(await client.recv())
                    assert metadata['language_conditioning'] == 'clip_film'
                    assert metadata['prompt_source'] == source
                    await client.send(packb(obs))
                    np.testing.assert_array_equal(unpackb(await client.recv())['action'], reference)
                    await client.send(packb({'reset': True}))
                    await client.send(packb(obs))
                    np.testing.assert_array_equal(unpackb(await client.recv())['action'], reference)
        asyncio.run(network())
    assert len(loaded) == 4 and len(encoder.calls) == 4


@pytest.mark.skipif(os.environ.get('ACT_TEST_REAL_CLIP') != '1', reason='Opt-in cached real CLIP CPU test')
def test_real_clip_short_standard_and_long_chunk_equivalence():
    from b1k_language import load_clip_text_encoder
    tokenizer, model = load_clip_text_encoder()
    for prompt in ['turning_on_radio', "Turn on the radio receiver that's on the table in the living room."]:
        with torch.inference_mode():
            expected = model(**tokenizer(prompt, return_tensors='pt', padding=True,
                                        return_attention_mask=True)).text_embeds[0]
        torch.testing.assert_close(encode_prompt(prompt, tokenizer, model), expected, atol=0, rtol=0)
    prompt = ' '.join(['Put the red object into the kitchen cabinet.'] * 20)
    content = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
    assert len(content) > 75
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(content), 75):
            ids = torch.tensor([[tokenizer.bos_token_id, *content[start:start + 75], tokenizer.eos_token_id]])
            embeddings.append(model(input_ids=ids, attention_mask=torch.ones_like(ids)).text_embeds[0])
    torch.testing.assert_close(encode_prompt(prompt, tokenizer, model), torch.stack(embeddings).mean(0),
                               atol=2e-6, rtol=2e-6)
