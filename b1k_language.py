"""Frozen, task-cached CLIP prompts for ACT visual FiLM conditioning."""

import json
from pathlib import Path

import torch


CLIP_MODEL = 'openai/clip-vit-large-patch14'
CLIP_REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
LANGUAGE_DIM = 768
PROMPT_SOURCES = ('task_name', 'task_description')
FILM_INITS = ('random', 'identity')  # random: nn.Linear default; identity: zero projection (beta = gamma = 0)
LONG_PROMPT_POLICY = 'mean_projected_75_token_chunks'


def language_mode(model_config):
    mode = model_config.get('language_conditioning', 'none')
    if mode not in ('none', 'clip_film'):
        raise ValueError(f'Unsupported language conditioning {mode}')
    if model_config.get('prompt_source', 'task_name') not in PROMPT_SOURCES:
        raise ValueError('Unsupported prompt source')
    if model_config.get('film_init', 'random') not in FILM_INITS:
        raise ValueError('Unsupported FiLM initialization')
    if mode != 'none' and model_config.get('policy_class', 'ACT') != 'ACT':
        raise ValueError('CLIP FiLM language conditioning is only supported for ACT, not CNNMLP')
    return mode


def task_prompts(root, task_map, source):
    if source not in PROMPT_SOURCES:
        raise ValueError(f'Unsupported prompt source {source}')
    if not task_map or any(type(i) is not int or not isinstance(name, str) or not name.strip()
                           for i, name in task_map.items()) or len(set(task_map.values())) != len(task_map):
        raise ValueError('Selected task map must contain unique integer IDs and nonempty names')
    if source == 'task_name':
        return dict(task_map)
    path = Path(root) / 'meta/tasks.jsonl'
    if not path.exists():
        raise ValueError(f'Task descriptions require {path}')
    selected_names = set(task_map.values())
    prompts = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid task JSON') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{line_number}: expected a task object')
            task_id, name = row.get('task_index'), row.get('task_name')
            if not isinstance(task_id, (int, str, float, type(None))) or not isinstance(name, (str, type(None))):
                raise ValueError(f'{path}:{line_number}: invalid task_index/task_name')
            if task_id not in task_map and name not in selected_names:
                continue
            if type(task_id) is not int or task_id not in task_map or task_map[task_id] != name:
                raise ValueError(f'{path}:{line_number}: conflicting selected task_index/task_name')
            if task_id in prompts:
                raise ValueError(f'{path}:{line_number}: duplicate selected task {task_id}')
            prompt = row.get('task')
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'{path}:{line_number}: missing selected {source} prompt for task {task_id}')
            prompts[task_id] = prompt
    missing = set(task_map) - prompts.keys()
    if missing:
        raise ValueError(f'{path}: missing selected prompts for task IDs {sorted(missing)}')
    return {i: prompts[i] for i in sorted(task_map)}


def load_clip_text_encoder():
    # Restore and unconditioned paths never import transformers or open its cache.
    try:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
    except ImportError as exc:
        raise RuntimeError('CLIP FiLM requires transformers; install requirements-b1k.txt') from exc
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP_MODEL, revision=CLIP_REVISION)
    model.requires_grad_(False).eval()
    return tokenizer, model


@torch.inference_mode()
def encode_prompt(prompt, tokenizer, model):
    content = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
    if len(content) <= 75:
        tokens = tokenizer(prompt, return_tensors='pt', padding=True, truncation=False,
                           return_attention_mask=True)
    else:
        chunks = [content[start:start + 75] for start in range(0, len(content), 75)]
        sequences = [[tokenizer.bos_token_id, *chunk, tokenizer.eos_token_id] for chunk in chunks]
        tokens = tokenizer.pad({'input_ids': sequences, 'attention_mask': [[1] * len(s) for s in sequences]},
                               padding=True, return_tensors='pt')
    if tokens['input_ids'].shape[1] > 77:
        raise ValueError('CLIP token sequence exceeds its 77-token context')
    device = next(model.parameters()).device
    tokens = {key: value.to(device) for key, value in tokens.items()}
    projected = model(**tokens).text_embeds
    if projected.ndim != 2 or projected.shape != (tokens['input_ids'].shape[0], LANGUAGE_DIM):
        raise ValueError(f'Expected CLIP projected embeddings with dimension {LANGUAGE_DIM}')
    if not torch.isfinite(projected).all():
        raise ValueError('Non-finite CLIP text embeddings')
    embedding = projected.float().mean(dim=0).cpu()
    if not torch.isfinite(embedding).all():
        raise ValueError('Non-finite mean CLIP text embedding')
    return embedding


def build_language_cache(root, task_map, source):
    prompts = task_prompts(root, task_map, source)
    tokenizer, model = load_clip_text_encoder()
    model.requires_grad_(False).eval()
    tasks = {i: {'task_name': task_map[i], 'prompt': prompt,
                 'embedding': encode_prompt(prompt, tokenizer, model).tolist()}
             for i, prompt in prompts.items()}
    return {'version': 1, 'model': CLIP_MODEL, 'revision': CLIP_REVISION, 'embedding_dim': LANGUAGE_DIM,
            'normalized': False, 'long_prompt_policy': LONG_PROMPT_POLICY, 'prompt_source': source,
            'tasks': tasks}


def language_embedding_table(model_config, task_map, cache):
    if language_mode(model_config) == 'none':
        if cache is not None:
            raise ValueError('Unconditioned checkpoint must not contain a language cache')
        return None
    expected = {'version': 1, 'model': CLIP_MODEL, 'revision': CLIP_REVISION, 'embedding_dim': LANGUAGE_DIM,
                'normalized': False, 'long_prompt_policy': LONG_PROMPT_POLICY,
                'prompt_source': model_config.get('prompt_source', 'task_name')}
    if not isinstance(cache, dict) or any(cache.get(key) != value for key, value in expected.items()):
        raise ValueError('Missing or incompatible checkpoint language cache model/revision/source/policy')
    tasks = cache.get('tasks')
    if not isinstance(tasks, dict) or set(tasks) != set(task_map) or not tasks:
        raise ValueError('Checkpoint language cache task map mismatch')
    embeddings = []
    for task_id in sorted(task_map):
        entry = tasks[task_id]
        if not isinstance(entry, dict) or entry.get('task_name') != task_map[task_id]:
            raise ValueError(f'Checkpoint language cache task name mismatch for {task_id}')
        prompt = entry.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f'Missing checkpoint language prompt for {task_id}')
        if expected['prompt_source'] == 'task_name' and prompt != task_map[task_id]:
            raise ValueError(f'Checkpoint task_name prompt mismatch for {task_id}')
        try:
            embedding = torch.tensor(entry.get('embedding'), dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f'Invalid checkpoint language embedding for {task_id}') from exc
        if embedding.shape != (LANGUAGE_DIM,) or not torch.isfinite(embedding).all():
            raise ValueError(f'Expected finite {LANGUAGE_DIM}d checkpoint language embedding for {task_id}')
        embeddings.append(embedding)
    return torch.stack(embeddings)


def validate_resume_prompts(root, task_map, cache):
    if cache is not None and (Path(root) / 'meta/tasks.jsonl').exists():
        prompts = task_prompts(root, task_map, cache['prompt_source'])
        if any(prompt != cache['tasks'][i]['prompt'] for i, prompt in prompts.items()):
            raise ValueError('Resume language prompts differ from checkpoint')


def language_for_qpos(qpos, embeddings):
    if embeddings is None:
        return None
    # The adapter appends one-hot categories in sorted task-ID order.
    return embeddings[qpos[:, -len(embeddings):].argmax(dim=-1)]
