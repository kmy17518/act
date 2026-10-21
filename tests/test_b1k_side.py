"""Side runs on the GB300 buildboxes: late goal fusion under DDP, --stop-file preemption, relative actions with
openpi pi05 quantile normalization, and validation L1 in action units."""

import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

from b1k_dataset import CAMERAS
from b1k_training import (committed_checkpoints, ddp_ignored_parameter_names, load_checkpoint, make_policy, parser,
                          train, unused_gradients)
from test_b1k import small_model_config, tiny_root  # noqa: F401  (fixture)


def late_config():
    """The buildbox2 recipe's late fusion (grid goal tokens through the shared backbone, zero goal tag, own goal LR),
    at test size, with a skipped decoder layer."""
    return dict(small_model_config(), state_dim=23, dec_layers=2, l1_normalization='valid', regime='image',
                task_onehot=False, goal_fusion='late', goal_views=[CAMERAS[0]], goal_tokens='grid',
                goal_encoder='shared_base', goal_tag_init='zero', goal_role_embedding=True, lr_goal=1e-4)


def late_batches(steps=3, per_rank=2, ranks=2):
    generator = torch.Generator().manual_seed(5)
    total = per_rank * ranks
    return [(torch.randn(total, 23, generator=generator), torch.rand(total, 3, 3, 32, 32, generator=generator),
             torch.randn(total, 4, 23, generator=generator), torch.zeros(total, 4, dtype=torch.bool),
             torch.rand(total, 1, 3, 32, 32, generator=generator)) for _ in range(steps)]


def late_ddp_worker(rank, world_size, store, result):
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.manual_seed(21)
    policy = make_policy(late_config(), 'cpu')
    optimizer = policy.configure_optimizers()
    zero_gradients = unused_gradients(policy)
    ignored = ddp_ignored_parameter_names(policy, zero_gradients)
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=rank, world_size=world_size)
    try:
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(policy, ignored)
        model = torch.nn.parallel.DistributedDataParallel(policy, broadcast_buffers=False, gradient_as_bucket_view=True)
        for step, (qpos, images, actions, pad, goal) in enumerate(late_batches()):
            part = slice(2 * rank, 2 * rank + 2)
            torch.manual_seed(100 * step + rank)  # the CVAE latent sample of this rank's half
            optimizer.zero_grad(set_to_none=True)
            model(qpos[part], images[part], actions[part], pad[part], goal=goal[part])['loss'].backward()
            for parameter, zeros in zero_gradients:
                parameter.grad = zeros
            optimizer.step()
    finally:
        dist.destroy_process_group()
    if rank == 0:
        torch.save({'ignored': ignored, 'state': policy.state_dict(),
                    'no_grad': [name for name, p in policy.named_parameters() if p.grad is None]}, result)


def test_late_fusion_trains_under_two_rank_ddp_like_one_process(tmp_path):
    """Late fusion's parameters all receive gradients except the padding head and the skipped decoder layers (the
    is_pad_head fix covers them), so two gloo ranks step without reducer errors and match one process that averages
    the two halves' gradients."""
    result = tmp_path / 'ddp.pt'
    mp.start_processes(late_ddp_worker, args=(2, str(tmp_path / 'store'), str(result)), nprocs=2, start_method='spawn')
    ddp = torch.load(result, weights_only=True)
    assert set(ddp['ignored']) == {name for name in ddp['state'] if name.startswith(
        ('model.is_pad_head.', 'model.transformer.decoder.layers.1.'))}
    assert set(ddp['no_grad']) == {'model.is_pad_head.weight', 'model.is_pad_head.bias'}
    torch.set_num_threads(1)
    torch.manual_seed(21)
    policy = make_policy(late_config(), 'cpu')
    optimizer = policy.configure_optimizers()
    assert [g['lr'] for g in optimizer.param_groups][-1] == 1e-4  # the goal group (goal tag)
    zero_gradients = unused_gradients(policy)
    for step, (qpos, images, actions, pad, goal) in enumerate(late_batches()):
        optimizer.zero_grad(set_to_none=True)
        for rank in range(2):
            part = slice(2 * rank, 2 * rank + 2)
            torch.manual_seed(100 * step + rank)
            (policy(qpos[part], images[part], actions[part], pad[part], goal=goal[part])['loss'] / 2).backward()
        for parameter, zeros in zero_gradients:
            parameter.grad = zeros
        optimizer.step()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(ddp['state'][name], value, atol=2e-6, rtol=1e-5, msg=name)
    assert ddp['state']['model.goal_role_embed.weight'].abs().sum() > 0  # the zero-initialized goal tag trains


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def train_rank_worker(rank, world_size, port, argvs, trigger, record):
    """One gloo rank of `train`; on rank 0, `trigger` = (stop file, check number) creates the stop file just before that
    stop check (check 1 runs before the first step, check k + 1 after step k). Writes this rank's stop checks."""
    import b1k_training
    os.environ.update(WORLD_SIZE=str(world_size), RANK=str(rank), LOCAL_RANK=str(rank), MASTER_ADDR='127.0.0.1',
                      MASTER_PORT=str(port))
    torch.set_num_threads(1)
    checks = []
    original = b1k_training.stop_requested

    def recorded(stop_file, device, distributed):
        if rank == 0 and trigger and len(checks) + 1 == trigger[1]:
            Path(trigger[0]).touch()
        checks.append(original(stop_file, device, distributed))
        return checks[-1]

    b1k_training.stop_requested = recorded
    returned = train(parser().parse_args(argvs[rank]))
    Path(record.format(rank=rank)).write_text(json.dumps({'checks': checks, 'returned': str(returned)}))


def run_ranks(tmp_path, argvs, trigger=None, name='run', timeout=600):
    """Two gloo ranks of `train`; ranks that disagree on when to stop deadlock in mismatched collectives and fail here."""
    record = str(tmp_path / f'{name}-rank{{rank}}.json')
    context = mp.start_processes(train_rank_worker, args=(2, free_port(), argvs, trigger, record), nprocs=2,
                                 start_method='spawn', join=False)
    deadline = time.monotonic() + timeout
    while not context.join(timeout=5):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.terminate()
            raise TimeoutError(f'{name}: the two ranks did not finish within {timeout} s')
    return [json.loads(Path(record.format(rank=rank)).read_text()) for rank in range(2)]


def test_stop_file_stops_both_gloo_ranks_at_the_same_step_and_leaves_a_resumable_checkpoint(tiny_root, tmp_path):
    output = tmp_path / 'run'
    stop = tmp_path / 'run.stop'  # rank 0's stop file; rank 1 watches a path that never exists
    common = ['--dataset-path', str(tiny_root), '--batch-size', '4', '--num-workers', '0', '--device', 'cpu',
              '--chunk-size', '4', '--image-size', '32', '32', '--hidden-dim', '32', '--dim-feedforward', '64',
              '--enc-layers', '1', '--dec-layers', '2', '--nheads', '4', '--no-pretrained-backbone', '--dropout', '0',
              '--regime', 'image', '--goal-fusion', 'late', '--goal-tokens', 'grid', '--goal-encoder', 'shared_base',
              '--goal-tag-init', 'zero', '--lr-goal', '1e-4', '--save-every', '5', '--export-every', '5',
              '--save-total-limit', '3', '--output-dir', str(output)]
    argvs = [common + ['--max-steps', '10', '--stop-file', str(path)] for path in (stop, tmp_path / 'never.stop')]
    ranks = run_ranks(tmp_path, argvs, trigger=(str(stop), 4), name='stopped')  # created after step 3
    assert ranks[0]['checks'] == ranks[1]['checks'] == [False, False, False, True]
    assert ranks[0]['returned'] == str(output / 'step_00000003.pt')
    assert [p.name for p in committed_checkpoints(output)] == ['step_00000003.pt']
    assert [p.name for p in committed_checkpoints(output / 'export_queue/full')] == ['step_00000003.pt']
    assert (output / 'export_queue/full/step_00000003.pt').stat().st_ino == (output / 'step_00000003.pt').stat().st_ino
    assert not (output / 'export_queue/eval').exists()  # off the 5-step grid: no eval-only export
    stopped = load_checkpoint(output / 'step_00000003.pt')
    assert stopped['checkpoint_type'] == 'full' and stopped['step'] == 3 and stopped['optimizer']['state']
    assert stopped['train_config']['stop_file'] == str(stop) and stopped['train_config']['world_size'] == 2
    assert [json.loads(line)['step'] for line in (output / 'metrics.jsonl').read_text().splitlines()] == [1, 2, 3]

    # a stop file present at launch: no step, no checkpoint
    ranks = run_ranks(tmp_path, [a + ['--resume', str(output / 'step_00000003.pt')] for a in argvs], name='at-launch')
    assert ranks[0]['checks'] == ranks[1]['checks'] == [True] and ranks[0]['returned'] == 'None'
    assert [p.name for p in committed_checkpoints(output)] == ['step_00000003.pt']

    stop.unlink()
    ranks = run_ranks(tmp_path, [a + ['--resume', str(output / 'step_00000003.pt')] for a in argvs], name='resumed')
    assert ranks[0]['checks'] == ranks[1]['checks'] == [False] * 8
    assert ranks[0]['returned'] == str(output / 'step_00000010.pt')
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == list(range(1, 11))
    assert all(np.isfinite(record['loss']) for record in records)
    assert [p.name for p in committed_checkpoints(output)] == ['step_00000003.pt', 'step_00000005.pt',
                                                              'step_00000010.pt']
    assert [p.name for p in committed_checkpoints(output / 'export_queue/full')] == ['step_00000010.pt']
    assert [p.name for p in committed_checkpoints(output / 'export_queue/eval')] == ['step_00000005.pt',
                                                                                    'step_00000010.pt']
    resumed = load_checkpoint(output / 'step_00000010.pt')
    assert resumed['step'] == 10 and resumed['train_config']['resume'] == str(output / 'step_00000003.pt')
    assert resumed['model_config']['goal_fusion'] == 'late'
