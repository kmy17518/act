"""Frozen, task-cached text prompts for ACT language conditioning (CLIP FiLM and the MT-ACT reproduction)."""

import json
from pathlib import Path

import torch


CLIP_MODEL = 'openai/clip-vit-large-patch14'
CLIP_REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
MINILM_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
MINILM_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MINILM_MAX_TOKENS = 256  # sentence_bert_config.json max_seq_length
LANGUAGE_DIM = 768  # CLIP; kept for callers that predate the encoder registry
PROMPT_SOURCES = ('task_name', 'task_description')
FILM_INITS = ('random', 'identity')  # random: nn.Linear default; identity: zero projection (beta = gamma = 0)
LONG_PROMPT_POLICY = 'mean_projected_75_token_chunks'
# Frozen text encoders. `clip`: CLIP ViT-L/14 projected text embeddings (768-d, unnormalized, long prompts
# averaged over 75-token chunks). `minilm`: sentence-transformers all-MiniLM-L6-v2 as RoboAgent's MT-ACT used
# it (masked mean pooling of the last hidden state, L2-normalized, 384-d, truncated to 256 word pieces).
ENCODERS = {
    'clip': {'model': CLIP_MODEL, 'revision': CLIP_REVISION, 'embedding_dim': 768, 'normalized': False,
             'long_prompt_policy': LONG_PROMPT_POLICY},
    'minilm': {'model': MINILM_MODEL, 'revision': MINILM_REVISION, 'embedding_dim': 384, 'normalized': True,
               'long_prompt_policy': f'truncate_{MINILM_MAX_TOKENS}_word_pieces_mean_pool_l2_normalize'},
}
# clip_film: FiLM after every ResNet block output (b1k.md). mt_act: RoboAgent's MT-ACT (FiLM inside the
# residual branch of ResNet stages 2-4, a shared learned text projection that also enters the transformer
# encoder as a token, CVAE style encoder over actions only, no one-hot task input; see detr_vae.py).
LANGUAGE_MODES = ('none', 'clip_film', 'mt_act')
BACKBONE_NORMS = ('frozen', 'batch')


def language_mode(model_config):
    mode = model_config.get('language_conditioning', 'none')
    if mode not in LANGUAGE_MODES:
        raise ValueError(f'Unsupported language conditioning {mode}')
    if model_config.get('prompt_source', 'task_name') not in PROMPT_SOURCES:
        raise ValueError('Unsupported prompt source')
    if model_config.get('film_init', 'random') not in FILM_INITS:
        raise ValueError('Unsupported FiLM initialization')
    if model_config.get('backbone_norm', 'frozen') not in BACKBONE_NORMS:
        raise ValueError('Unsupported backbone normalization')
    if mode != 'none' and language_encoder(model_config) not in ENCODERS:
        raise ValueError('Unsupported language encoder')
    if mode != 'none' and model_config.get('policy_class', 'ACT') != 'ACT':
        raise ValueError('Language conditioning is only supported for ACT, not CNNMLP')
    return mode


def language_encoder(model_config):
    """Encoder name; checkpoints that predate the registry (clip_film only) used CLIP."""
    return model_config.get('language_encoder', 'clip')


def language_dim(model_config):
    return ENCODERS[language_encoder(model_config)]['embedding_dim'] if language_mode(model_config) != 'none' else None


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


def load_minilm_encoder():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError('MiniLM prompts require transformers; install requirements-b1k.txt') from exc
    tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL, revision=MINILM_REVISION)
    model = AutoModel.from_pretrained(MINILM_MODEL, revision=MINILM_REVISION)
    model.requires_grad_(False).eval()
    return tokenizer, model


@torch.inference_mode()
def encode_prompt_minilm(prompt, tokenizer, model):
    """sentence-transformers all-MiniLM-L6-v2 semantics: truncate, masked mean pooling, L2 normalization."""
    tokens = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=MINILM_MAX_TOKENS,
                       return_attention_mask=True)
    device = next(model.parameters()).device
    tokens = {key: value.to(device) for key, value in tokens.items()}
    hidden = model(**tokens).last_hidden_state
    dim = ENCODERS['minilm']['embedding_dim']
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != dim:
        raise ValueError(f'Expected MiniLM hidden states with dimension {dim}')
    mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
    embedding = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)[0].cpu()
    if not torch.isfinite(embedding).all():
        raise ValueError('Non-finite MiniLM text embedding')
    return embedding


def build_language_cache(root, task_map, source, encoder='clip'):
    if encoder not in ENCODERS:
        raise ValueError(f'Unsupported language encoder {encoder}')
    prompts = task_prompts(root, task_map, source)
    if encoder == 'clip':
        tokenizer, model = load_clip_text_encoder()
        encode = encode_prompt
    else:
        tokenizer, model = load_minilm_encoder()
        encode = encode_prompt_minilm
    model.requires_grad_(False).eval()
    tasks = {i: {'task_name': task_map[i], 'prompt': prompt, 'embedding': encode(prompt, tokenizer, model).tolist()}
             for i, prompt in prompts.items()}
    return {'version': 1, **ENCODERS[encoder], 'prompt_source': source, 'tasks': tasks}


def language_embedding_table(model_config, task_map, cache):
    if language_mode(model_config) == 'none':
        if cache is not None:
            raise ValueError('Unconditioned checkpoint must not contain a language cache')
        return None
    expected = {'version': 1, **ENCODERS[language_encoder(model_config)],
                'prompt_source': model_config.get('prompt_source', 'task_name')}
    dim = expected['embedding_dim']
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
        if embedding.shape != (dim,) or not torch.isfinite(embedding).all():
            raise ValueError(f'Expected finite {dim}d checkpoint language embedding for {task_id}')
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
