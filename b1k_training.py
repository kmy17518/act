"""Step-based training and standalone checkpoints for upstream ACT and CNNMLP policies."""

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
import fcntl
from functools import partial
import json
import logging
import os
from pathlib import Path
import random
import re
import tempfile
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from b1k_dataset import B1KDataset, CAMERAS, OBS_KEYS, STATE_INDICES, VIDEO_KEYS, SplitBatchSampler, StepBatchSampler
from b1k_frame_cache import dequantize_images
from detr.models.transformer import use_fused_attention
from policy import ACTPolicy, CNNMLPPolicy


LOGGER = logging.getLogger(__name__)
CHECKPOINT_VERSION = 1


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'act-b1k' or checkpoint.get('version') != CHECKPOINT_VERSION:
        raise ValueError('Expected a standalone ACT/CNNMLP B1K checkpoint')
    adapter = checkpoint['adapter_config']
    if adapter['state_indices'] != STATE_INDICES or adapter['action_dim'] != 23:
        raise ValueError('Unsupported checkpoint robot mapping')
    if adapter['video_keys'] != VIDEO_KEYS or adapter['observation_keys'] != OBS_KEYS:
        raise ValueError('Unsupported checkpoint camera mapping')
    if adapter['action_transform'] != 'identity' or adapter['action_shift'] != 0:
        raise ValueError('Unsupported checkpoint action semantics')
    name = policy_class(checkpoint['model_config'])
    if name == 'CNNMLP' and checkpoint['model_config'].get('image_size', [480, 640]) != adapter['image_size']:
        raise ValueError('CNNMLP checkpoint model/adapter image sizes differ')
    return checkpoint


def policy_class(model_config):
    name = model_config.get('policy_class', 'ACT')
    if name not in ('ACT', 'CNNMLP'):
        raise ValueError(f'Unsupported policy class {name}')
    if name == 'CNNMLP' and model_config.get('num_queries', 1) != 1:
        raise ValueError('CNNMLP predicts one action, not an action chunk')
    return name


def make_policy(model_config, device, restoring=False, fused_optimizer=False, skip_unused_decoder_layers=True,
                attention='auto'):
    """Build the upstream policy.

    `fused_optimizer`, `skip_unused_decoder_layers` and `attention` are runtime execution choices, not
    saved model configuration. ACT consumes only the first stacked decoder output, so with the skip
    enabled the remaining decoder layers are not executed; predictions and every gradient are unchanged
    (see `unused_gradients` for the optimizer side). `attention` selects the nn.MultiheadAttention path
    (`detr.models.transformer.use_fused_attention`).
    """
    config = dict(model_config, programmatic=True, device=str(device), fused_optimizer=fused_optimizer)
    if restoring:
        config['pretrained_backbone'] = False
    policy_type = ACTPolicy if policy_class(config) == 'ACT' else CNNMLPPolicy
    policy = policy_type(config)
    if skip_unused_decoder_layers and hasattr(policy.model, 'decoder_layers_used'):
        policy.model.decoder_layers_used = 1
    use_fused_attention(attention)  # validate the mode
    for module in policy.modules():
        if hasattr(module, 'attention'):
            module.attention = attention
    return policy


def unused_gradients(policy):
    """Persistent zero gradients for parameters the forward pass never reaches.

    Upstream autograd produces exactly-zero gradients for those parameters, and AdamW still applies
    its decoupled weight decay to them. Assigning the same zeros after each backward keeps every
    parameter and optimizer-state trajectory identical while their forward/backward work is skipped.
    """
    parameters = getattr(policy.model, 'unused_parameters', list)()
    return [(parameter, torch.zeros_like(parameter)) for parameter in parameters]


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path, writer, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            temporary.replace(path)
        else:
            os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save(checkpoint, path):
    _atomic_write(path, lambda stream: torch.save(checkpoint, stream))


def atomic_json(value, path):
    _atomic_write(path, lambda stream: stream.write(json.dumps(value, indent=2).encode()), replace=True)


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another trainer holds {output / "run.lock"}') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def committed_checkpoints(directory):
    return sorted((path for path in directory.glob('step_*.pt')
                   if re.fullmatch(r'step_\d{8,}\.pt', path.name) and path.is_file() and not path.is_symlink()),
                  key=lambda path: int(path.stem[5:]))


def prune_checkpoints(directory, limit):
    if limit > 0:
        for path in committed_checkpoints(directory)[:-limit]:
            path.unlink(missing_ok=True)
        _sync_directory(directory)


def save_full_checkpoint(saved, output, save_total_limit=0, export_queue=False):
    path = output / f'step_{saved["step"]:08d}.pt'
    atomic_save(saved, path)
    if export_queue:
        queue = output / 'export_queue/full'
        queue.mkdir(parents=True, exist_ok=True)
        os.link(path, queue / path.name)
        _sync_directory(queue)
    link = output / 'latest.tmp'
    link.unlink(missing_ok=True)
    link.symlink_to(path.name)
    link.replace(output / 'latest.pt')
    _sync_directory(output)
    prune_checkpoints(output, save_total_limit)
    if export_queue:
        prune_checkpoints(queue, 1)
    return path


def save_eval_checkpoint(run, policy, step, output):
    saved = {'format': 'act-b1k', 'version': CHECKPOINT_VERSION, 'checkpoint_type': 'eval',
             'step': step, **run,
             'model': {key: value.detach().cpu() for key, value in policy.state_dict().items()}}
    path = output / 'export_queue/eval' / f'step_{step:08d}.pt'
    if path.exists():
        previous = load_checkpoint(path)
        same_metadata = all(previous[key] == saved[key] for key in
                            ('step', 'model_config', 'adapter_config', 'task_map', 'normalization'))
        same_model = previous['model'].keys() == saved['model'].keys() and all(
            torch.equal(previous['model'][key], value) for key, value in saved['model'].items())
        if not same_metadata or not same_model:
            raise FileExistsError(f'Existing eval export differs at step {step}; use another output directory')
        return path
    atomic_save(saved, path)
    return path


def optimizer_batches(loader, batch_size, loader_batch_size, device=None, stream=None):
    """Yield one ordered optimizer batch per step: (images, qpos, actions, is_pad, per-sample timings).

    Worker slices are reassembled in sampler order. With a CUDA `device`, the four tensors are
    allocated on the device and each pinned slice is copied straight into place (`stream`, when given,
    carries the copies so they overlap the previous step's compute; consume with `consume_batch`).
    Without a device the batch is assembled on the host exactly as before.
    """
    on_device = device is not None and torch.device(device).type == 'cuda'
    context = torch.cuda.stream(stream) if on_device and stream is not None else nullcontext()

    def allocate(first):
        if on_device:
            return torch.empty((batch_size, *first.shape[1:]), dtype=first.dtype, device=device)
        return torch.empty((batch_size, *first.shape[1:]), dtype=first.dtype, pin_memory=loader.pin_memory)

    if not loader_batch_size or loader_batch_size >= batch_size:
        for batch in loader:
            if on_device:
                with context:
                    tensors = [value.to(device, non_blocking=True) for value in batch[:4]]
                yield (*tensors, batch[4])
                del tensors
            else:
                yield batch
            del batch
        return
    slices = []
    parts = (batch_size + loader_batch_size - 1) // loader_batch_size
    for batch in loader:
        slices.append(batch)
        if len(slices) == parts:
            tensors = []
            with context:
                for column in range(4):
                    merged = allocate(slices[0][column])
                    offset = 0
                    for part in slices:
                        value = part[column]
                        merged[offset:offset + len(value)].copy_(value, non_blocking=on_device)
                        offset += len(value)
                    if offset != batch_size:
                        raise RuntimeError('Incorrect optimizer batch size from data loader')
                    tensors.append(merged)
            timings = {key: torch.cat([part[4][key] for part in slices]) for key in slices[0][4]}
            slices.clear()
            yield (*tensors, timings)
            del tensors, timings, merged, value, part, batch
    if slices:
        raise RuntimeError('Incomplete optimizer batch from data loader')


def compile_policy(policy, mode):
    """torch.compile the network forward in place; parameters, state_dict keys and the optimizer are untouched."""
    policy.model.forward = torch.compile(policy.model.forward, mode=mode)
    return policy


def match_optimizer_state_layout(optimizer):
    """Give restored moment tensors the memory layout of their (possibly channels-last) parameters.

    Restored states are plain contiguous tensors; fused/foreach kernels pair elements positionally, so
    the layouts must agree. `empty_like` preserves the parameter's strides and `copy_` is a logical copy.
    """
    for group in optimizer.param_groups:
        for parameter in group['params']:
            for key, value in list(optimizer.state.get(parameter, {}).items()):
                if torch.is_tensor(value) and value.shape == parameter.shape and value.stride() != parameter.stride():
                    optimizer.state[parameter][key] = torch.empty_like(parameter).copy_(value)


def prepare_images(images, channels_last=True):
    """Loader images to the float (batch, camera, 3, H, W) tensor in [0, 1] the policies expect.

    uint8 frame-cache batches arrive as (batch, camera, H, W, 3) and become float images whose
    per-camera slices are dense channels-last blocks; float batches only get re-laid out the same
    way when `channels_last` asks for it.
    """
    if images.dtype == torch.uint8:
        images = dequantize_images(images)
        return images if channels_last else images.contiguous()
    if channels_last and images.dim() == 5:
        return images.permute(1, 0, 3, 4, 2).contiguous().permute(1, 0, 4, 2, 3)
    return images


def consume_batch(batch, stream=None, channels_last=True):
    """Hand a prefetched batch to the current stream and convert its images for the forward pass."""
    images, qpos, actions, is_pad, timings = batch
    if stream is not None:
        current = torch.cuda.current_stream(images.device)
        current.wait_stream(stream)
        for tensor in (images, qpos, actions, is_pad):
            tensor.record_stream(current)
    return prepare_images(images, channels_last), qpos, actions, is_pad, timings


def configure_cpu_threads(worker_id=None, torch_threads=1, arrow_threads=1, opencv_threads=1):
    import pyarrow as pa
    torch.set_num_threads(torch_threads)
    if worker_id is not None:
        torch.set_num_interop_threads(torch_threads)
    pa.set_cpu_count(arrow_threads)
    pa.set_io_thread_count(arrow_threads)
    try:
        import cv2
    except ImportError:
        return
    cv2.setNumThreads(opencv_threads)


@contextmanager
def wandb_run(args, output, checkpoint):
    identity_path = output / 'wandb.json'
    local = json.loads(identity_path.read_text()) if identity_path.exists() else None
    saved = checkpoint.get('wandb') if checkpoint else None
    if saved and local and any(saved.get(key) != local.get(key) for key in ('id', 'project', 'entity')):
        raise ValueError('Output W&B identity differs from the resumed checkpoint')
    saved = saved or local
    for key in ('id', 'project', 'entity', 'name'):
        requested = getattr(args, f'wandb_{key}')
        if saved and requested is not None and requested != saved.get(key):
            raise ValueError(f'--wandb-{key} differs from the saved run; resume must preserve W&B identity')
    if args.wandb_mode == 'disabled':
        yield None, saved
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError('W&B requested but not installed; install requirements-b1k.txt') from exc
    identity = dict(saved or {'id': args.wandb_id or uuid.uuid4().hex[:8],
                             'project': args.wandb_project or 'act-b1k',
                             'entity': args.wandb_entity, 'name': args.wandb_name or output.name})
    identity['mode'] = args.wandb_mode
    if args.wandb_mode == 'online':
        key = os.environ.get('WANDB_API_KEY')
        if not key:
            raise RuntimeError('Online W&B requires WANDB_API_KEY; refusing offline fallback')
        if not wandb.login(verify=True):
            raise RuntimeError('Online W&B authentication failed; refusing offline fallback')
    atomic_json(identity, identity_path)
    tracked = wandb.init(project=identity['project'], entity=identity['entity'], name=identity['name'],
                         id=identity['id'], mode=args.wandb_mode, dir=str(output),
                         resume='must' if saved and saved.get('mode') == 'online' and checkpoint
                         and args.wandb_mode == 'online' else 'allow')
    if tracked is None:
        raise RuntimeError('W&B did not create a run')
    try:
        if args.wandb_mode == 'online' and tracked.settings.mode != 'online':
            raise RuntimeError('Online W&B initialization returned a non-online run; refusing fallback')
        if tracked.id != identity['id']:
            raise RuntimeError('W&B initialization changed the requested run ID; refusing a new run')
        for key in ('id', 'project', 'entity', 'name'):
            identity[key] = getattr(tracked, key)
        atomic_json(identity, identity_path)
        tracked.define_metric('step')
        tracked.define_metric('*', step_metric='step')
        yield tracked, identity
    except BaseException:
        tracked.finish(exit_code=1)
        raise
    else:
        tracked.finish()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', required=True)
    p.add_argument('--task-names', nargs='+', help='Default: every complete local task')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--max-steps', type=int, default=50000, help='Total optimizer steps, including resumed steps')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--loader-batch-size', type=int, help='Worker slice size; reassembled before one full-batch update')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--prefetch-factor', type=int, default=1)
    p.add_argument('--torch-threads', type=int, default=None, help='Main-process CPU threads; default keeps PyTorch setting')
    p.add_argument('--worker-threads', type=int, default=1)
    p.add_argument('--arrow-threads', type=int, default=1)
    p.add_argument('--opencv-threads', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', type=Path)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--policy-class', choices=['ACT', 'CNNMLP'], default='ACT')
    p.add_argument('--chunk-size', type=int, default=100, help='ACT prediction length; CNNMLP always uses one action')
    p.add_argument('--image-size', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'),
                   help='Default: ACT 240 240; CNNMLP 480 640 (original convolution/flatten path)')
    p.add_argument('--position-embedding', choices=['sine', 'learned'], default='sine')
    p.add_argument('--pre-norm', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--hidden-dim', type=int, default=512)
    p.add_argument('--dim-feedforward', type=int, default=3200)
    p.add_argument('--enc-layers', type=int, default=4)
    p.add_argument('--dec-layers', type=int, default=7)
    p.add_argument('--nheads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--kl-weight', type=float, default=10)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lr-backbone', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--pretrained-backbone', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--save-first-step', action='store_true', help='Also save and queue a full checkpoint at step 1')
    p.add_argument('--save-total-limit', type=int, default=0, help='Retain newest N local full checkpoints; 0 keeps all')
    p.add_argument('--export-every', type=int, default=0, help='Queue eval-only checkpoints every N steps; 0 disables queues')
    p.add_argument('--wandb-project')
    p.add_argument('--wandb-entity')
    p.add_argument('--wandb-name')
    p.add_argument('--wandb-id')
    p.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='disabled')
    p.add_argument('--stats-max-frames', type=int, help='Smoke only: use first N selected frames, not full statistics')
    p.add_argument('--cache-row-groups', type=int, default=2)
    p.add_argument('--timestamp-tolerance', type=float, default=0.008)
    p.add_argument('--frame-cache', type=Path, help='Root of the uint8 resized-frame cache built by '
                   'scripts/b1k/build_frame_cache.py for this image size; default decodes video per sample')
    p.add_argument('--matmul-precision', choices=['highest', 'high', 'medium'], default='highest',
                   help='torch.set_float32_matmul_precision for fp32 matmuls; "high" enables TF32 tensor cores '
                        '(convolutions already default to TF32 in PyTorch)')
    p.add_argument('--autocast', choices=['none', 'bf16'], default='none',
                   help='bf16 autocast for the forward pass on CUDA; weights, optimizer state, output heads, '
                        'latent distribution and losses stay fp32')
    p.add_argument('--channels-last', action=argparse.BooleanOptionalAction, default=True,
                   help='Channels-last (NHWC) memory layout for the convolutional backbone (layout only)')
    p.add_argument('--fused-optimizer', action=argparse.BooleanOptionalAction, default=True,
                   help='Single-kernel fused AdamW on CUDA (same update rule)')
    p.add_argument('--cudnn-benchmark', action=argparse.BooleanOptionalAction, default=True,
                   help='Let cuDNN autotune convolution algorithms for the fixed batch shape')
    p.add_argument('--compile', choices=['none', 'default', 'max-autotune-no-cudagraphs'], default='none',
                   help='torch.compile the policy network (CUDA only)')
    p.add_argument('--compute-unused-decoder-layers', action='store_true',
                   help='Also run the ACT decoder layers whose outputs are discarded (upstream behavior); '
                        'by default they are skipped and receive the same exactly-zero gradients')
    p.add_argument('--attention', choices=['auto', 'fused', 'explicit'], default='auto',
                   help='nn.MultiheadAttention path: explicit upstream matmul+softmax, fused scaled-dot-product '
                        'kernels, or auto (fused only under autocast)')
    return p


def train(args):
    if min(args.max_steps, args.batch_size, args.save_every) < 1 or args.num_workers < 0:
        raise ValueError('Steps, batch size and save interval must be positive; workers must be nonnegative')
    if args.loader_batch_size is not None and args.loader_batch_size < 1:
        raise ValueError('--loader-batch-size must be positive')
    if min(args.prefetch_factor, args.worker_threads, args.arrow_threads, args.opencv_threads) < 1:
        raise ValueError('Prefetch factor and CPU thread caps must be positive')
    if args.torch_threads is not None and args.torch_threads < 1:
        raise ValueError('--torch-threads must be positive')
    if min(args.save_total_limit, args.export_every) < 0:
        raise ValueError('Retention limit and export interval must be nonnegative')
    if args.stats_max_frames is not None and args.stats_max_frames < 2:
        raise ValueError('--stats-max-frames must be at least 2')
    output = Path(args.output_dir).resolve()
    root = Path(args.dataset_path).resolve()
    if output == root or root in output.parents:
        raise ValueError('--output-dir cannot be inside the read-only dataset')
    with run_lock(output), ExitStack() as resources:
        return _train(args, output, root, resources)


def _train(args, output, root, resources):
    if not args.resume and ((output / 'run.json').exists() or any(output.glob('step_*.pt'))):
        raise FileExistsError(f'{output} already contains a run; use --resume or another output directory')
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint and (checkpoint.get('checkpoint_type') == 'eval' or 'optimizer' not in checkpoint):
        raise ValueError('Cannot resume training from an eval-only checkpoint; use a full step checkpoint or latest.pt')
    if checkpoint:
        if args.max_steps <= checkpoint['step']:
            raise ValueError(f'--max-steps must exceed resumed step {checkpoint["step"]}')
        conflicts = [p for p in committed_checkpoints(output) if checkpoint['step'] < int(p.stem[5:])]
        if conflicts:
            raise FileExistsError(f'Resume would overwrite existing snapshots: {conflicts}; choose another output directory')
        LOGGER.info('Resuming %s at optimizer step %d; saved architecture/preprocessing/seed are authoritative',
                    args.resume, checkpoint['step'])
        adapter = checkpoint['adapter_config']
        model_config = dict(checkpoint['model_config'])
        model_config.setdefault('policy_class', 'ACT')
        task_names = args.task_names or list(checkpoint['task_map'].values())
        args.seed = checkpoint['train_config']['seed']
    else:
        image_size = args.image_size or ([240, 240] if args.policy_class == 'ACT' else [480, 640])
        if args.policy_class == 'CNNMLP' and (args.pre_norm or args.position_embedding != 'sine'):
            raise ValueError('Position embeddings and pre-norm are ACT architecture options; CNNMLP does not use them')
        if min(image_size) < 1:
            raise ValueError('--image-size dimensions must be positive')
        if args.policy_class == 'CNNMLP' and min(image_size) < 385:
            raise ValueError('CNNMLP requires images at least 385x385; default is 480x640')
        adapter = {'image_size': list(image_size), 'state_indices': STATE_INDICES,
                   'video_keys': VIDEO_KEYS, 'observation_keys': OBS_KEYS, 'action_dim': 23,
                   'action_transform': 'identity', 'action_shift': 0, 'task_conditioning': 'onehot',
                   'timestamp_tolerance': args.timestamp_tolerance, 'image_resize': 'bilinear_antialias',
                   'image_normalization': f'rgb_div255_then_imagenet_in_{args.policy_class}Policy'}
        task_names = args.task_names
        model_config = {'policy_class': args.policy_class,
                        'num_queries': args.chunk_size if args.policy_class == 'ACT' else 1,
                        'hidden_dim': args.hidden_dim,
                        'dim_feedforward': args.dim_feedforward, 'enc_layers': args.enc_layers,
                        'dec_layers': args.dec_layers, 'nheads': args.nheads, 'dropout': args.dropout,
                        'kl_weight': args.kl_weight, 'lr': args.lr, 'lr_backbone': args.lr_backbone,
                        'weight_decay': args.weight_decay, 'backbone': 'resnet18',
                        'pretrained_backbone': args.pretrained_backbone, 'camera_names': CAMERAS,
                        'position_embedding': args.position_embedding, 'pre_norm': args.pre_norm,
                        'dilation': False, 'masks': False, 'action_dim': 23}
        if args.policy_class == 'CNNMLP':
            model_config['image_size'] = list(image_size)
    if policy_class(model_config) == 'ACT':
        if model_config['hidden_dim'] % 4 or model_config['hidden_dim'] % model_config['nheads']:
            raise ValueError('hidden-dim must be divisible by four and nheads')
    tracked, wandb_identity = resources.enter_context(wandb_run(args, output, checkpoint))
    configure_cpu_threads(torch_threads=args.torch_threads or torch.get_num_threads(),
                          arrow_threads=args.arrow_threads, opencv_threads=args.opencv_threads)
    dataset = B1KDataset(root, task_names, model_config['num_queries'], adapter['image_size'],
                         cache_row_groups=args.cache_row_groups, profile_reads=True,
                         timestamp_tolerance=adapter['timestamp_tolerance'],
                         frame_cache=args.frame_cache.resolve() if args.frame_cache else None)
    resources.callback(dataset.close)
    if checkpoint and dataset.task_map != checkpoint['task_map']:
        raise ValueError('Resume task map differs from checkpoint')
    model_config['state_dim'] = len(STATE_INDICES) + len(dataset.task_map)
    fingerprint = dataset.fingerprint()
    output.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        stats = checkpoint['normalization']
        if stats['fingerprint'] != fingerprint:
            raise ValueError('Resume dataset metadata/files differ from checkpoint; use the same local subset')
    else:
        stats_path = output / f'stats_{fingerprint[:16]}_{args.stats_max_frames or "all"}.json'
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
        else:
            stats = dataset.compute_stats(args.stats_max_frames)
            atomic_json(stats, stats_path)
    if args.stats_max_frames is None and (stats['approximate'] or stats['count'] != len(dataset)):
        raise ValueError('Full-task statistics required; approximate checkpoint statistics need explicit '
                         '--stats-max-frames for smoke runs only')
    if stats['approximate']:
        LOGGER.warning('SMOKE statistics only: %d/%d frames; not full-dataset normalization', stats['count'], len(dataset))
    dataset.stats = stats
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    torch.set_float32_matmul_precision(args.matmul_precision)
    if cuda:
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    policy = make_policy(model_config, device, restoring=checkpoint is not None,
                         fused_optimizer=args.fused_optimizer and cuda,
                         skip_unused_decoder_layers=not args.compute_unused_decoder_layers, attention=args.attention)
    if args.channels_last:
        policy.to(memory_format=torch.channels_last)  # only 4-D (convolution) weights change layout
    optimizer = policy.configure_optimizers()
    zero_gradients = unused_gradients(policy)
    start = 0
    if checkpoint:
        policy.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        match_optimizer_state_layout(optimizer)
        start = checkpoint['step']
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint['cuda_rng'] is not None:
            torch.cuda.set_rng_state(checkpoint['cuda_rng'], device)
    if args.max_steps <= start:
        raise ValueError(f'--max-steps must exceed resumed step {start}')
    train_config = vars(args).copy()
    train_config['resume'] = str(args.resume) if args.resume else None
    train_config['frame_cache'] = str(args.frame_cache) if args.frame_cache else None
    run = {'model_config': model_config, 'adapter_config': adapter, 'train_config': train_config,
           'task_map': dataset.task_map, 'normalization': stats, 'wandb': wandb_identity}
    atomic_json(run, output / ('resume_run.json' if checkpoint else 'run.json'))
    if tracked:
        tracked.config.update(run, allow_val_change=True)
    sampler = StepBatchSampler(dataset, args.batch_size, start, args.max_steps, args.seed)
    loader_kwargs = {'num_workers': args.num_workers, 'pin_memory': device.type == 'cuda',
                     'generator': torch.Generator().manual_seed(args.seed)}
    if args.num_workers:
        loader_kwargs.update(multiprocessing_context='spawn', prefetch_factor=args.prefetch_factor,
                             worker_init_fn=partial(configure_cpu_threads, torch_threads=args.worker_threads,
                                                     arrow_threads=args.arrow_threads,
                                                     opencv_threads=args.opencv_threads))
    loader_sampler = SplitBatchSampler(sampler, args.loader_batch_size) if args.loader_batch_size else sampler
    loader = DataLoader(dataset, batch_sampler=loader_sampler, **loader_kwargs)
    if args.compile != 'none' and cuda:
        compile_policy(policy, args.compile)
    policy.train()
    autocast = ((lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if args.autocast == 'bf16' and cuda
                else nullcontext)
    copy_stream = torch.cuda.Stream(device) if cuda else None
    LOGGER.info('Training %d -> %d steps on %s with real upstream %s (matmul %s, autocast %s, attention %s, '
                'channels_last %s, fused optimizer %s, compile %s, frame cache %s, skipped unused decoder parameters %d)',
                start, args.max_steps, device, policy_class(model_config), args.matmul_precision, args.autocast,
                args.attention, args.channels_last, args.fused_optimizer and cuda, args.compile, bool(args.frame_cache),
                sum(parameter.numel() for parameter, _ in zero_gradients))
    begin = time.monotonic()
    previous_end = begin
    batches = optimizer_batches(loader, args.batch_size, args.loader_batch_size, device=device, stream=copy_stream)
    fetch_start = time.monotonic()
    upcoming = next(batches, None)
    upcoming_wait = time.monotonic() - fetch_start
    step = start
    with (output / 'metrics.jsonl').open('a') as metrics:
        while upcoming is not None:
            step += 1
            batch, data_wait = upcoming, upcoming_wait
            batch_ready = time.monotonic()
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            images, qpos, actions, is_pad, timings = consume_batch(batch, copy_stream, args.channels_last)
            with autocast():
                losses = policy(qpos, images, actions, is_pad)
            losses['loss'].backward()
            for parameter, zeros in zero_gradients:
                parameter.grad = zeros
            # The next batch is assembled on the device while this step's forward/backward kernels run.
            fetch_start = time.monotonic()
            upcoming = next(batches, None)
            upcoming_wait = time.monotonic() - fetch_start
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError(f'Non-finite loss at step {step}')
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            record = {'step': step, **{k: float(v.detach()) for k, v in losses.items()},
                      'grad_norm': float(grad_norm), 'lr': optimizer.param_groups[0]['lr'],
                      **{key: float(value.mean()) for key, value in timings.items()}}
            if cuda:
                torch.cuda.synchronize(device)
                record.update({'gpu/allocated_bytes': torch.cuda.memory_allocated(device),
                               'gpu/reserved_bytes': torch.cuda.memory_reserved(device),
                               'gpu/peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                               'gpu/peak_reserved_bytes': torch.cuda.max_memory_reserved(device)})
            else:
                record.update({f'gpu/{key}_bytes': 0 for key in
                               ('allocated', 'reserved', 'peak_allocated', 'peak_reserved')})
            trained = time.monotonic()
            if args.export_every and step % args.export_every == 0:
                exported = save_eval_checkpoint(run, policy, step, output)
                LOGGER.info('Queued eval-only checkpoint %s', exported)
            if step % args.save_every == 0 or step == args.max_steps or (args.save_first_step and step == 1):
                saved = {'format': 'act-b1k', 'version': CHECKPOINT_VERSION, 'checkpoint_type': 'full',
                         'step': step, **run, 'model': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                         'torch_rng': torch.get_rng_state(),
                         'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
                path = save_full_checkpoint(saved, output, args.save_total_limit, bool(args.export_every))
                LOGGER.info('Saved standalone checkpoint %s', path)
            finished = time.monotonic()
            # data_wait_s: host time blocked fetching this step's batch (overlapped with the previous
            # step's compute after step 1); train_s covers this step's launch, compute and that overlap.
            record.update({'elapsed_s': finished - begin, 'timing/data_wait_s': data_wait,
                           'timing/train_s': trained - batch_ready, 'timing/checkpoint_s': finished - trained,
                           'timing/step_s': finished - previous_end,
                           'samples_per_s': args.batch_size / (finished - previous_end)})
            metrics.write(json.dumps(record) + '\n')
            metrics.flush()
            LOGGER.info('%s', json.dumps(record))
            if tracked:
                tracked.log(record, step=step)
            del images, qpos, actions, is_pad, losses, grad_norm, batch, timings
            previous_end = time.monotonic()
    return path


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    train(parser().parse_args())
