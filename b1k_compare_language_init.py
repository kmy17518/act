"""Paired, short ACT initialization diagnostics; never resumes production training."""

import argparse
from collections import deque
from contextlib import contextmanager, ExitStack
import copy
from functools import partial
import hashlib
from itertools import chain
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from b1k_dataset import B1KDataset, CAMERAS, OBS_KEYS, STATE_INDICES, VIDEO_KEYS, StepBatchSampler
from b1k_language import language_embedding_table, language_for_qpos
from b1k_training import (CHECKPOINT_VERSION, atomic_json, atomic_save, configure_cpu_threads,
                          make_policy, run_lock)
from detr.models.backbone import FiLMLayer


LOGGER = logging.getLogger(__name__)
ARMS = ('baseline', 'random_film', 'identity_film')
INITIALIZATION_SEED = 0
DEFAULT_CACHE_RUN = Path(__file__).resolve().parent / (
    'outputs/turning-on-radio-act-clipfilm-taskname-bs1560-300k-20260916/run.json')
IMAGE_SIZE = (240, 240)


def model_config(state_dim):
    return {'policy_class': 'ACT', 'num_queries': 100, 'hidden_dim': 512,
            'dim_feedforward': 3200, 'enc_layers': 4, 'dec_layers': 7, 'nheads': 8,
            'dropout': 0.1, 'kl_weight': 10.0, 'lr': 1e-5, 'lr_backbone': 1e-5,
            'weight_decay': 1e-4, 'backbone': 'resnet18', 'pretrained_backbone': True,
            'camera_names': CAMERAS, 'position_embedding': 'sine', 'pre_norm': False,
            'language_conditioning': 'none', 'prompt_source': 'task_name',
            'dilation': False, 'masks': False, 'action_dim': 23, 'state_dim': state_dim}


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(tensor_bytes(value))
    return digest.hexdigest()


def film_names(policy):
    return {f'{name}.{key}' for name, module in policy.named_modules() if isinstance(module, FiLMLayer)
            for key in module.state_dict()}


def prove_shared_state(policies):
    reference = policies['baseline'].state_dict()
    if film_names(policies['baseline']):
        raise AssertionError('Baseline must not contain FiLM')
    hashes = {}
    for arm, policy in policies.items():
        state = policy.state_dict()
        extra = set(state) - set(reference)
        if extra != film_names(policy) or set(reference) - set(state):
            raise AssertionError(f'{arm}: non-FiLM state differs from baseline schema')
        for name, value in reference.items():
            other = state[name]
            if value.shape != other.shape or value.dtype != other.dtype or tensor_bytes(value) != tensor_bytes(other):
                raise AssertionError(f'{arm}: shared state is not byte-exact: {name}')
        hashes[arm] = state_hash({name: state[name] for name in reference})
    if len(set(hashes.values())) != 1:
        raise AssertionError('Initial shared state hashes differ')
    return {'byte_exact': True, 'shared_hashes': hashes, 'shared_state_keys': sorted(reference),
            'shared_parameter_tensors': len(list(policies['baseline'].named_parameters())),
            'shared_buffer_tensors': len(list(policies['baseline'].named_buffers()))}


def optimizer_proof(policies):
    seen = set()
    report = {}
    for arm, policy in policies.items():
        optimizer = policy.configure_optimizers()
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        expected = {id(p) for p in policy.parameters() if p.requires_grad}
        ids = [id(p) for p in parameters]
        if len(ids) != len(set(ids)) or set(ids) != expected or seen.intersection(ids):
            raise AssertionError('Optimizers must own every trainable parameter exactly once, independently')
        if not isinstance(optimizer, torch.optim.AdamW):
            raise AssertionError('Expected upstream AdamW')
        seen.update(ids)
        report[arm] = {'every_trainable_parameter_once': True, 'independent': True,
                       'film_parameter_tensors': len(film_names(policy)),
                       'groups': [{'lr': group['lr'], 'weight_decay': group['weight_decay'],
                                   'betas': list(group['betas']), 'eps': group['eps'],
                                   'parameter_tensors': len(group['params'])}
                                  for group in optimizer.param_groups]}
    return report


def build_arms(config, device):
    """Construct baseline once; all non-FiLM state comes from that one initialization."""
    torch.manual_seed(INITIALIZATION_SEED)
    baseline = make_policy(dict(config, language_conditioning='none'), device)
    shared = baseline.state_dict()
    policies = {'baseline': baseline}
    for arm in ARMS[1:]:
        torch.manual_seed(INITIALIZATION_SEED)
        policy = make_policy(dict(config, language_conditioning='clip_film'), device, restoring=True)
        mismatch = policy.load_state_dict(shared, strict=False)
        if mismatch.unexpected_keys or set(mismatch.missing_keys) != film_names(policy):
            raise AssertionError(f'{arm}: incomplete shared initialization copy')
        if arm == 'identity_film':
            with torch.no_grad():
                for module in policy.modules():
                    if isinstance(module, FiLMLayer):
                        module.lang_proj.weight.zero_()
                        module.lang_proj.bias.zero_()
        policies[arm] = policy
    proof = prove_shared_state(policies)
    proof['optimizer'] = optimizer_proof(policies)
    proof['initialization_seed'] = INITIALIZATION_SEED
    proof['film_hashes'] = {arm: state_hash({name: value for name, value in policy.state_dict().items()
                                           if name in film_names(policy)})
                             for arm, policy in policies.items()}
    proof['identity_film_zero'] = all(not torch.count_nonzero(value) for name, value in
                                    policies['identity_film'].state_dict().items()
                                    if name in film_names(policies['identity_film']))
    if not proof['identity_film_zero'] or proof['film_hashes']['random_film'] == proof['film_hashes']['identity_film']:
        raise AssertionError('Expected zero identity FiLM and distinct random FiLM')
    return policies, proof


def rng_snapshot(device):
    return {'cpu': torch.get_rng_state().clone(),
            'cuda': torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None}


def restore_rng(snapshot, device):
    torch.set_rng_state(snapshot['cpu'])
    if snapshot['cuda'] is not None:
        torch.cuda.set_rng_state(snapshot['cuda'], device)


def step_rng(seed, step, device):
    value = seed + step + 1
    torch.manual_seed(value)
    return value, rng_snapshot(device)


def rng_hash(snapshot):
    return state_hash({key: value for key, value in snapshot.items() if value is not None})


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def device_batch(batch, device):
    return tuple(value.to(device, non_blocking=True) for value in batch[:4])


def policy_kwargs(arm, qpos, embeddings):
    return {} if arm == 'baseline' else {'lang_emb': language_for_qpos(qpos, embeddings)}


def check_step_zero(policies, batch, embeddings, seed, device, rtol=2e-5, atol=2e-6):
    images, qpos, actions, padding = batch
    caller_rng = rng_snapshot(device)
    modes = {arm: policy.training for arm, policy in policies.items()}
    seed_value, snapshot = step_rng(seed, 0, device)
    losses, predictions, after = {}, {}, {}
    try:
        for arm, policy in policies.items():
            policy.eval()
            restore_rng(snapshot, device)
            with torch.no_grad():
                predictions[arm] = policy(qpos, images, **policy_kwargs(arm, qpos, embeddings)).detach().cpu()
            policy.train()
            restore_rng(snapshot, device)
            values = policy(qpos, images, actions, padding, **policy_kwargs(arm, qpos, embeddings))
            losses[arm] = {key: float(value.detach()) for key, value in values.items()}
            after[arm] = rng_hash(rng_snapshot(device))
            del values
        torch.testing.assert_close(predictions['baseline'], predictions['identity_film'], rtol=rtol, atol=atol)
        for key in losses['baseline']:
            torch.testing.assert_close(torch.tensor(losses['baseline'][key]),
                                       torch.tensor(losses['identity_film'][key]), rtol=rtol, atol=atol)
        if len(set(after.values())) != 1:
            raise AssertionError('Arms consumed different forward RNG streams')
    finally:
        for arm, policy in policies.items():
            policy.train(modes[arm])
        restore_rng(caller_rng, device)
    return {'passed': True, 'seed': seed_value, 'rtol': rtol, 'atol': atol,
            'train_losses': losses, 'same_forward_rng_consumption': True, 'rng_after_hashes': after,
            'identity_inference_max_abs_error': float((predictions['baseline'] - predictions['identity_film']).abs().max()),
            'identity_train_loss_abs_errors': {key: abs(losses['baseline'][key] - losses['identity_film'][key])
                                               for key in losses['baseline']},
            'random_inference_max_abs_difference': float((predictions['baseline'] - predictions['random_film']).abs().max()),
            'batch_hash': state_hash(dict(zip(('images', 'qpos', 'actions', 'padding'), batch)))}


def train_step(policies, batch, embeddings, seed, step, device):
    images, qpos, actions, padding = batch
    value, snapshot = step_rng(seed, step, device)
    records, after = {}, {}
    for arm, policy in policies.items():
        policy.train()
        optimizer = policy.configure_optimizers()
        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        begin = time.monotonic()
        restore_rng(snapshot, device)
        losses = policy(qpos, images, actions, padding, **policy_kwargs(arm, qpos, embeddings))
        after[arm] = rng_hash(rng_snapshot(device))
        if not all(torch.isfinite(loss) for loss in losses.values()):
            raise FloatingPointError(f'{arm}: non-finite loss at step {step}')
        synchronize(device)
        forwarded = time.monotonic()
        losses['loss'].backward()
        gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'), error_if_nonfinite=True)
        names = film_names(policy)
        film_parameters = [p for name, p in policy.named_parameters() if name in names]
        if any(p.grad is None for p in film_parameters):
            raise AssertionError(f'{arm}: missing FiLM gradient')
        film_gradient = (torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in film_parameters]))
                         if film_parameters else torch.tensor(0.0))
        synchronize(device)
        backwarded = time.monotonic()
        optimizer.step()
        synchronize(device)
        finished = time.monotonic()
        records[arm] = {'kind': 'train', 'arm': arm, 'step': step, 'rng_seed': value,
                        'l1': float(losses['l1'].detach()), 'kl': float(losses['kl'].detach()),
                        'total': float(losses['loss'].detach()), 'loss': float(losses['loss'].detach()),
                        'grad_norm': float(gradient),
                        'film_grad_norm': float(film_gradient),
                        'timing/forward_s': forwarded - begin, 'timing/backward_s': backwarded - forwarded,
                        'timing/optimizer_s': finished - backwarded, 'timing/train_s': finished - begin}
        optimizer.zero_grad(set_to_none=True)
        del losses
    if len(set(after.values())) != 1:
        raise AssertionError(f'Step {step}: arms consumed different forward RNG streams')
    return records


def evaluate(policies, batches, embeddings, device, step):
    """Held-out teacher-free inference with the upstream all-zero latent."""
    snapshot = rng_snapshot(device)
    modes = {arm: policy.training for arm, policy in policies.items()}
    sums = {arm: [0.0, 0, 0, 0.0] for arm in policies}
    try:
        with torch.no_grad():
            for cpu_batch in batches:
                images, qpos, actions, padding = device_batch(cpu_batch, device)
                for arm, policy in policies.items():
                    policy.eval()
                    restore_rng(snapshot, device)
                    synchronize(device)
                    begin = time.monotonic()
                    prediction = policy(qpos, images, **policy_kwargs(arm, qpos, embeddings))
                    error = ((prediction - actions).abs() * ~padding.unsqueeze(-1)).sum()
                    if not torch.isfinite(error):
                        raise FloatingPointError(f'{arm}: non-finite evaluation at step {step}')
                    sums[arm][0] += float(error)
                    sums[arm][1] += int((~padding).sum()) * actions.shape[-1]
                    sums[arm][2] += actions.numel()
                    synchronize(device)
                    sums[arm][3] += time.monotonic() - begin
    finally:
        for arm, policy in policies.items():
            policy.train(modes[arm])
        restore_rng(snapshot, device)
    if any(valid == 0 for _, valid, _, _ in sums.values()):
        raise ValueError('Evaluation requires non-padding targets')
    return {arm: {'kind': 'eval', 'arm': arm, 'step': step, 'inference_l1': error / valid, 'l1': error / valid,
                   'inference_l1_act_denominator': error / total, 'valid_action_values': valid,
                   'timing/eval_s': elapsed}
            for arm, (error, valid, total, elapsed) in sums.items()}


def restrict_episodes(dataset, episode_ids):
    """Rebuild all frame-index bookkeeping before statistics or sampling."""
    episode_ids = set(episode_ids)
    if not episode_ids or not episode_ids.issubset(dataset.by_id):
        raise ValueError('Episode restriction must be a nonempty known subset')
    dataset.close()
    dataset.episodes = [ep for ep in dataset.episodes if ep['episode_index'] in episode_ids]
    dataset.by_id = {int(ep['episode_index']): ep for ep in dataset.episodes}
    dataset.lengths = np.array([ep['length'] for ep in dataset.episodes], dtype=np.int64)
    dataset.ends = dataset.lengths.cumsum()
    dataset.starts = dataset.ends - dataset.lengths
    dataset.positions = {int(ep['episode_index']): i for i, ep in enumerate(dataset.episodes)}
    dataset._table = dataset._frame_rows = None  # in-memory rows (frame-cache mode) are indexed by position
    dataset.stats = None
    return dataset


def split_dataset(dataset, every=10):
    if every < 2:
        raise ValueError('Holdout stride must be at least two')
    ids = sorted(dataset.by_id)
    heldout = ids[every - 1::every]
    if not heldout:
        raise ValueError('Not enough episodes for a held-out split')
    training = [episode_id for episode_id in ids if episode_id not in set(heldout)]
    evaluation = copy.copy(dataset)
    evaluation.__dict__.update(dataset.__getstate__())
    restrict_episodes(evaluation, heldout)
    restrict_episodes(dataset, training)
    return evaluation, {'rule': f'every {every}th sorted episode (1-based)',
                         'train_episode_ids': training, 'eval_episode_ids': heldout,
                         'train_frames': len(dataset), 'eval_frames': len(evaluation),
                         'normalization_scope': 'train_episodes_only'}


def fixed_eval_samples(dataset, count, seed):
    rng = np.random.default_rng(seed)
    episodes = dataset.episodes
    counts = [count // len(episodes) + (i < count % len(episodes)) for i in range(len(episodes))]
    frames = [rng.choice(ep['length'], size=n, replace=n > ep['length']).tolist()
              for ep, n in zip(episodes, counts)]
    return [{'episode_id': int(episodes[i % len(episodes)]['episode_index']),
             'frame': int(frames[i % len(episodes)][i // len(episodes)])} for i in range(count)]


def cached_language(path, task_map):
    raw = Path(path).read_bytes()
    run = json.loads(raw)
    source_map = {int(key): value for key, value in run['task_map'].items()}
    if source_map != task_map:
        raise ValueError('Cached CLIP task map differs from selected dataset')
    cache = copy.deepcopy(run['language_cache'])
    cache['tasks'] = {int(key): value for key, value in cache['tasks'].items()}
    table = language_embedding_table({'language_conditioning': 'clip_film', 'prompt_source': 'task_name'},
                                     task_map, cache).detach()
    return run, cache, table, hashlib.sha256(raw).hexdigest()


def training_stats(dataset, path):
    fingerprint = dataset.fingerprint()
    path = Path(path)
    if path.exists():
        stats = json.loads(path.read_text())
    else:
        stats = dataset.compute_stats()
        atomic_json(stats, path)
    if (stats['fingerprint'] != fingerprint or stats['approximate'] or stats['count'] != len(dataset)
            or stats['std_correction'] != 1 or stats['std_floor'] != 0.01):
        raise ValueError('Exact training-only normalization cache does not match the split')
    for prefix, size in [('qpos', len(STATE_INDICES)), ('action', 23)]:
        mean, std = np.asarray(stats[f'{prefix}_mean']), np.asarray(stats[f'{prefix}_std'])
        if mean.shape != (size,) or std.shape != (size,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Invalid normalization values')
    dataset.stats = stats
    return stats


def summarize(history):
    return {arm: {'final100_count': len(records),
                   'final100_mean': {key: float(np.mean([row[key] for row in records]))
                                     for key in ('l1', 'kl', 'total', 'grad_norm', 'film_grad_norm')}}
            for arm, records in history.items()}


@contextmanager
def comparison_wandb(args, output, manifest, source):
    if args.wandb_mode == 'disabled':
        yield {}, {}
        return
    import wandb
    if args.wandb_mode == 'online' and not os.environ.get('WANDB_API_KEY'):
        raise RuntimeError('Online W&B requires WANDB_API_KEY')
    group = args.wandb_group or output.name
    runs, identities = {}, {}
    try:
        for arm in ARMS:
            identity = {'id': uuid.uuid4().hex[:12], 'name': f'{args.wandb_name}-{arm}', 'group': group,
                        'project': args.wandb_project or source.get('wandb', {}).get('project', 'act-b1k'),
                        'entity': args.wandb_entity or source.get('wandb', {}).get('entity'),
                        'mode': args.wandb_mode}
            run = wandb.init(**identity, dir=str(output), resume='never', reinit='create_new',
                             job_type='controlled-initialization', config={'arm': arm, **manifest},
                             settings=wandb.Settings(console='off', disable_code=True))
            if run is None:
                raise RuntimeError('W&B did not create a run')
            runs[arm] = run
            if run.settings.mode != args.wandb_mode or run.id != identity['id']:
                raise RuntimeError('W&B mode or identity mismatch')
            run.define_metric('step')
            run.define_metric('*', step_metric='step')
            identities[arm] = dict(identity, entity=run.entity)
        atomic_json(identities, output / 'wandb.json')
        yield runs, identities
    except BaseException:
        for run in runs.values():
            run.finish(exit_code=1)
        raise
    else:
        for run in runs.values():
            run.finish()


def write_records(stream, records):
    for record in records.values():
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    stream.flush()


def write_manifest(manifest, output):
    atomic_json(manifest, output / 'run.json')
    atomic_json(manifest, output / 'manifest.json')


def save_checkpoints(policies, config, adapter, cache, manifest, output, step):
    paths = {}
    for arm, policy in policies.items():
        path = output / arm / 'final.pt'
        saved = {'format': 'act-b1k', 'version': CHECKPOINT_VERSION, 'checkpoint_type': 'full',
                 'step': step, 'comparison_arm': arm, 'model_config': dict(config, language_conditioning=
                    'none' if arm == 'baseline' else 'clip_film'),
                 'adapter_config': adapter, 'task_map': manifest['task_map'],
                 'normalization': manifest['normalization'], 'train_config': manifest['train_config'],
                 'comparison': {'initialization': manifest['initialization'], 'split': manifest['split']},
                 'wandb': manifest['wandb'].get(arm),
                 'model': {key: value.detach().cpu() for key, value in policy.state_dict().items()},
                 'optimizer': policy.configure_optimizers().state_dict(),
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state(next(policy.parameters()).device)
                             if next(policy.parameters()).is_cuda else None}
        if arm != 'baseline':
            saved['language_cache'] = cache
        atomic_save(saved, path)
        paths[arm] = str(path)
    return paths


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-path', default='/tmp/dev/datasets/2026-challenge-demos')
    p.add_argument('--task-names', nargs='+', default=['turning_on_radio'])
    p.add_argument('--expected-episodes', type=int, default=200)
    p.add_argument('--cache-run', type=Path, default=DEFAULT_CACHE_RUN, help='Existing real CLIP cache in run.json')
    p.add_argument('--stats-cache', type=Path, help='Optional reusable TRAIN-only exact statistics JSON')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0, help='Sampling/training RNG seed; shared initialization is always seed 0')
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--eval-samples', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--holdout-every', type=int, default=10)
    p.add_argument('--torch-threads', type=int, default=2)
    p.add_argument('--parity-rtol', type=float, default=2e-5)
    p.add_argument('--parity-atol', type=float, default=2e-6)
    p.add_argument('--wandb-mode', choices=['disabled', 'offline', 'online'], default='disabled')
    p.add_argument('--wandb-project')
    p.add_argument('--wandb-entity')
    p.add_argument('--wandb-name', default='act-init-20260917')
    p.add_argument('--wandb-group')
    return p


def resolve_device(value):
    device = torch.device(value)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only CPU and CUDA are supported')
    return device


def compare(args):
    if min(args.max_steps, args.batch_size, args.eval_every, args.eval_samples,
           args.eval_batch_size, args.torch_threads, args.expected_episodes) < 1 or args.num_workers < 0:
        raise ValueError('Steps, batches, intervals and counts must be positive; workers nonnegative')
    if args.holdout_every < 2 or args.seed < 0 or args.seed + args.max_steps + 1 >= 2**63 or args.eval_seed < 0:
        raise ValueError('Invalid split or RNG seed')
    if not 0 <= args.parity_atol <= 2e-6 or not 0 <= args.parity_rtol <= 2e-5:
        raise ValueError('Parity tolerance must remain tight')
    output, root = Path(args.output_dir).resolve(), Path(args.dataset_path).resolve()
    writes = [output] + ([args.stats_cache.resolve()] if args.stats_cache else [])
    if any(Path('/tmp') not in path.parents or root == path or root in path.parents for path in writes):
        raise ValueError('Outputs/caches must be under /tmp and outside the read-only dataset')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Comparison output must be a new empty directory; no resume or overwrite')
    device = resolve_device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    configure_cpu_threads(torch_threads=args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed % 2**32)
    with run_lock(output), ExitStack() as resources:
        return _compare(args, output, root, device, resources)


def _compare(args, output, root, device, resources):
    config = model_config(len(STATE_INDICES) + len(args.task_names))
    dataset = B1KDataset(root, args.task_names, config['num_queries'], IMAGE_SIZE, profile_reads=True)
    resources.callback(dataset.close)
    if len(dataset.episodes) != args.expected_episodes:
        raise ValueError(f'Expected {args.expected_episodes} complete episodes, found {len(dataset.episodes)}')
    source, cache, embeddings, cache_hash = cached_language(args.cache_run, dataset.task_map)
    config['state_dim'] = len(STATE_INDICES) + len(dataset.task_map)
    full_fingerprint = dataset.fingerprint()
    evaluation, split = split_dataset(dataset, args.holdout_every)
    resources.callback(evaluation.close)
    stats_path = args.stats_cache or output / f'stats_train_{dataset.fingerprint()[:16]}.json'
    stats = training_stats(dataset, stats_path)
    evaluation.stats = stats
    samples = fixed_eval_samples(evaluation, args.eval_samples, args.eval_seed)
    eval_batches = [default_collate([evaluation.sample_at(row['episode_id'], row['frame'])
                                     for row in samples[start:start + args.eval_batch_size]])
                    for start in range(0, len(samples), args.eval_batch_size)]
    evaluation.close()
    adapter = {'image_size': list(IMAGE_SIZE), 'state_indices': STATE_INDICES, 'video_keys': VIDEO_KEYS,
               'observation_keys': OBS_KEYS, 'action_dim': 23, 'action_transform': 'identity', 'action_shift': 0,
               'task_conditioning': 'onehot', 'timestamp_tolerance': dataset.timestamp_tolerance,
               'image_resize': 'bilinear_antialias', 'image_normalization': 'rgb_div255_then_imagenet_in_ACTPolicy'}
    policies, proof = build_arms(config, device)
    embeddings = embeddings.to(device)
    train_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                            capture_output=True, text=True, check=False).stdout.strip()
    manifest = {'format': 'act-initialization-comparison', 'version': 1, 'status': 'initialized',
                'model_config': config, 'adapter_config': adapter, 'train_config': train_config,
                'task_map': dataset.task_map, 'normalization': stats, 'language_cache': cache,
                'initialization': proof, 'split': split, 'full_dataset_fingerprint': full_fingerprint,
                'eval_samples': samples, 'eval_seed': args.eval_seed, 'eval_latent': 'all_zeros',
                'eval_metric': 'normalized_action_L1_valid_values; ACT padded denominator also logged',
                'eval_is_rollout': False, 'production_resume_supported': False,
                'checkpoint_usage': 'standalone inference or custom diagnostics only; do not resume production training',
                'cache_run': str(args.cache_run.resolve()), 'cache_run_sha256': cache_hash,
                'stats_cache': str(Path(stats_path).resolve()), 'clip_encoder_loaded': False,
                'precision': 'float32', 'autocast': False, 'gradient_accumulation_steps': 1,
                'gradient_clipping': None, 'tf32': False, 'optimizer': 'independent upstream AdamW',
                'rng_pairing': 'CPU and selected CUDA RNG restored before EVERY arm forward; seed + step + 1',
                'batch_pairing': 'one StepBatchSampler/DataLoader; same tensors passed to all arms',
                'arm_order': list(ARMS), 'long_run_batch_size': source.get('train_config', {}).get('batch_size'),
                'diagnostic_batch_size': args.batch_size, 'git_commit': commit,
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
                'device': str(device), 'wandb': {}}
    write_manifest(manifest, output)
    sampler = StepBatchSampler(dataset, args.batch_size, 0, args.max_steps, args.seed)
    loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': device.type == 'cuda',
                     'generator': torch.Generator().manual_seed(args.seed)}
    if args.num_workers:
        loader_kwargs.update(multiprocessing_context='spawn', prefetch_factor=1,
                             worker_init_fn=partial(configure_cpu_threads, torch_threads=1))
    loader = DataLoader(dataset, batch_sampler=sampler, **loader_kwargs)
    begin = time.monotonic()
    iterator = iter(loader)
    first = next(iterator)
    probe = device_batch(first, device)
    manifest['step_zero'] = check_step_zero(policies, probe, embeddings, args.seed, device,
                                            args.parity_rtol, args.parity_atol)
    del probe
    manifest['status'] = 'step_zero_verified'
    write_manifest(manifest, output)
    tracked, identities = resources.enter_context(comparison_wandb(args, output, manifest, source))
    manifest['wandb'] = identities
    write_manifest(manifest, output)
    history = {arm: deque(maxlen=100) for arm in ARMS}
    with (output / 'metrics.jsonl').open('x') as metrics, (output / 'eval.jsonl').open('x') as eval_metrics:
        initial_eval = evaluate(policies, eval_batches, embeddings, device, 0)
        write_records(eval_metrics, initial_eval)
        for arm, run in tracked.items():
            run.log({'step': 0, **{f'eval/{key}': value for key, value in initial_eval[arm].items()
                                   if key not in ('kind', 'arm', 'step')}}, step=0)
        previous = time.monotonic()
        final_eval = initial_eval
        for step, batch in enumerate(chain([first], iterator), 1):
            ready = time.monotonic()
            transferred = device_batch(batch, device)
            records = train_step(policies, transferred, embeddings, args.seed, step, device)
            del transferred
            for arm, record in records.items():
                record.update({'timing/data_wait_s': ready - previous, 'elapsed_s': time.monotonic() - begin})
                if len(batch) > 4:
                    record.update({key: float(value.mean()) for key, value in batch[4].items()})
                history[arm].append(record)
            write_records(metrics, records)
            eval_records = {}
            if step % args.eval_every == 0 or step == args.max_steps:
                eval_records = evaluate(policies, eval_batches, embeddings, device, step)
                write_records(eval_metrics, eval_records)
                final_eval = eval_records
            for arm, run in tracked.items():
                logged = {'step': step, **{f'train/{key}': value for key, value in records[arm].items()
                                          if key not in ('kind', 'arm', 'step')}}
                if eval_records:
                    logged.update({f'eval/{key}': value for key, value in eval_records[arm].items()
                                   if key not in ('kind', 'arm', 'step')})
                run.log(logged, step=step)
            if step == 1 or step % 10 == 0:
                LOGGER.info('step=%d totals=%s', step, {arm: row['total'] for arm, row in records.items()})
            previous = time.monotonic()
    paths = save_checkpoints(policies, config, adapter, cache, manifest, output, args.max_steps)
    summary = {'max_steps': args.max_steps, 'elapsed_s': time.monotonic() - begin,
               'arms': summarize(history), 'initial_eval': initial_eval, 'final_eval': final_eval,
               'checkpoints': paths, 'step_zero_passed': True}
    atomic_json(summary, output / 'summary.json')
    manifest.update(status='completed', summary='summary.json', checkpoints=paths)
    write_manifest(manifest, output)
    for arm, run in tracked.items():
        run.summary.update(summary['arms'][arm])
        run.summary['final_inference_l1'] = final_eval[arm]['inference_l1']
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for name in ('httpx', 'httpcore', 'urllib3', 'huggingface_hub'):
        logging.getLogger(name).setLevel(logging.ERROR)
    compare(parser().parse_args())
