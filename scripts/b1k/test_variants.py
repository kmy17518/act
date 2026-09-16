#!/usr/bin/env python3
"""Reproducible native-policy train/resume and dataset-free WebSocket coverage matrix."""

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# These are branch-coverage cases, not a sweep over training hyperparameters.
MATRIX = [
    {'name': 'act-sine-post', 'policy': 'ACT', 'position': 'sine', 'pre_norm': False, 'image_size': [64, 64]},
    {'name': 'act-learned-post', 'policy': 'ACT', 'position': 'learned', 'pre_norm': False, 'image_size': [64, 64]},
    {'name': 'act-sine-pre', 'policy': 'ACT', 'position': 'sine', 'pre_norm': True, 'image_size': [64, 64]},
    {'name': 'act-learned-pre', 'policy': 'ACT', 'position': 'learned', 'pre_norm': True, 'image_size': [64, 64]},
    {'name': 'cnnmlp-native', 'policy': 'CNNMLP', 'image_size': [480, 640]},
    {'name': 'cnnmlp-spatial', 'policy': 'CNNMLP', 'image_size': [416, 416]},
]


def device_environment(device):
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return
    selected = os.environ.get('CUDA_VISIBLE_DEVICES')
    if selected is None or not selected.isdigit():
        raise RuntimeError('CUDA requires explicit CUDA_VISIBLE_DEVICES=<one idle physical GPU index>')
    processes = subprocess.check_output(
        ['nvidia-smi', '-i', selected, '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    memory = subprocess.check_output(
        ['nvidia-smi', '-i', selected, '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
        text=True).strip()
    used, utilization = map(int, memory.split(','))
    if processes or used > 100 or utilization != 0:
        raise RuntimeError(f'Refusing occupied GPU {selected}: processes={processes!r}, memory/utilization={memory}')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_observations(root, tasks, path):
    import numpy as np
    from b1k_dataset import B1KDataset, OBS_KEYS
    dataset = B1KDataset(root, tasks, chunk_size=4)
    arrays, sources = {}, []
    try:
        for task in dataset.task_map:
            episode = next(ep for ep in dataset.episodes if ep['task_index'] == task)
            for frame in (0, min(1, episode['length'] - 1)):
                images, state, _, _, task_id = dataset.raw_sample(episode['episode_index'], frame)
                i = len(sources)
                arrays[f'{i}_state'] = state
                arrays[f'{i}_task'] = np.array(task_id, dtype=np.int64)
                for cam, image in enumerate(images):
                    arrays[f'{i}_cam{cam}'] = image
                sources.append({'episode_index': episode['episode_index'], 'frame': frame, 'task_id': task_id,
                                'camera_shapes': [list(image.shape) for image in images]})
        np.savez(path, **arrays)
    finally:
        dataset.close()
    return sources


def read_observations(path):
    import numpy as np
    from b1k_dataset import OBS_KEYS
    with np.load(path) as arrays:
        count = len(arrays.files) // 5
        return [{'robot_r1::proprio': arrays[f'{i}_state'][None],
                 'task_id': int(arrays[f'{i}_task']),
                 **{key: arrays[f'{i}_cam{cam}'][None] for cam, key in enumerate(OBS_KEYS)}}
                for i in range(count)]


async def exercise_websocket(checkpoint, predictor, observations, temporal_agg, horizon):
    import numpy as np
    import torch
    import websockets.asyncio.client
    import websockets.asyncio.server
    from websockets.exceptions import ConnectionClosedError
    from b1k_dataset import OBS_KEYS, preprocess_image, preprocess_state
    from b1k_server import B1KServer, packb, unpackb

    first, second = observations[:2]
    stats, tasks = checkpoint['normalization'], checkpoint['task_map']
    def predict(obs):
        qpos = torch.stack([preprocess_state(state, obs['task_id'], stats, tasks)
                            for state in obs['robot_r1::proprio']])
        images = torch.stack([torch.stack([preprocess_image(image, checkpoint['adapter_config']['image_size'])
                                          for image in obs[key]]) for key in OBS_KEYS], dim=1)
        return predictor(qpos, images)

    reference = [predict(obs) for obs in observations[:2]]
    is_cnn = checkpoint['model_config']['policy_class'] == 'CNNMLP'
    mode = 'single_action' if is_cnn else ('temporal_aggregation' if temporal_agg else 'chunked')
    server = B1KServer(checkpoint, predictor, action_horizon=horizon, temporal_agg=temporal_agg)
    checks, responses = [], []
    async with websockets.asyncio.server.serve(server.handler, '127.0.0.1', 0, process_request=server.health,
                                               max_size=64 * 1024 * 1024, compression=None) as running:
        port = running.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection('127.0.0.1', port)
        writer.write(b'GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
        await writer.drain()
        assert (await reader.read()).endswith(b'OK\n')
        writer.close()
        await writer.wait_closed()
        checks.append('health')
        url = f'ws://127.0.0.1:{port}'
        async with websockets.asyncio.client.connect(url, max_size=64 * 1024 * 1024) as a, \
                   websockets.asyncio.client.connect(url, max_size=64 * 1024 * 1024) as b:
            metadata = unpackb(await a.recv())
            assert metadata == unpackb(await b.recv())
            assert metadata['policy'] == checkpoint['model_config']['policy_class']
            assert metadata['execution_mode'] == mode
            if is_cnn:
                assert metadata['chunk_size'] == metadata['action_horizon'] == 1
            checks.append('checkpoint_dispatch_metadata')
            async def request(client, obs):
                await client.send(packb(obs))
                action = unpackb(await asyncio.wait_for(client.recv(), 120))['action']
                assert action.shape == (len(obs['robot_r1::proprio']), 23)
                assert action.dtype == np.float32 and np.isfinite(action).all()
                responses.append({'shape': list(action.shape), 'min': float(action.min()), 'max': float(action.max())})
                return action

            initial = await request(a, first)
            np.testing.assert_allclose(initial, reference[0] if is_cnn else reference[0][:, 0], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(await request(b, first), initial)
            for step in range(1, 6):
                actual = await request(a, observations[step % 2])
                if is_cnn:
                    expected = reference[step % 2]
                elif temporal_agg:
                    candidates = np.stack([reference[i % 2][0, step - i]
                                           for i in range(max(0, step - 3), step + 1)])
                    weights = np.exp(-.01 * np.arange(len(candidates)))
                    expected = (candidates * (weights / weights.sum())[:, None]).sum(0)[None]
                else:
                    plan_step = step - step % horizon
                    expected = reference[plan_step % 2][:, step % horizon]
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
            checks.append('independent_upstream_execution_reference')
            await a.send(packb({'reset': True}))
            try:
                await asyncio.wait_for(a.recv(), .05)
                raise AssertionError('Reset sent acknowledgement')
            except asyncio.TimeoutError:
                pass
            np.testing.assert_array_equal(await request(a, first), initial)
            b_second = await request(b, second)
            if temporal_agg:
                weights = np.exp(-.01 * np.arange(2))
                expected = (reference[0][:, 1] * weights[0] + reference[1][:, 0] * weights[1]) / weights.sum()
            else:
                expected = reference[1] if is_cnn else reference[0][:, 1]
            np.testing.assert_allclose(b_second, expected, rtol=2e-5, atol=2e-5)
            checks.extend(['reset_no_ack', 'connection_isolation'])
            batch = {key: np.repeat(value, 2, axis=0) if isinstance(value, np.ndarray) else value
                     for key, value in first.items()}
            await request(a, batch)
            task_ids = list(tasks)
            if len(task_ids) > 1:
                batch['task_id'] = np.array(task_ids[:2], dtype=np.int64)
                await request(a, batch)
                batch['task_id'] = batch['task_id'][::-1].copy()
                await request(a, batch)
            await request(a, first)
            for obs in observations[2:]:
                await request(a, obs)
            checks.extend(['batch_resize', 'per_slot_task_change', 'real_observations_both_tasks', 'finite_float32_B23'])
            await a.send(packb(dict(first, task_id=999999)))
            try:
                await a.recv()
                raise AssertionError('Unknown task accepted')
            except ConnectionClosedError as exc:
                assert exc.rcvd.code == 1008
            checks.append('unknown_task_rejected')
    return {'result': 'PASS', 'mode': mode, 'metadata': metadata, 'checks': checks, 'responses': responses}


def serving_worker(args):
    import torch
    from b1k_training import load_checkpoint
    from b1k_server import PolicyPredictor
    torch.set_num_threads(args.threads)
    checkpoint = load_checkpoint(args.checkpoint)
    dataset_root = str(Path(checkpoint['train_config']['dataset_path']).resolve())
    def deny_dataset(event, values):
        if event in ('open', 'os.listdir', 'os.scandir') and values:
            path = values[0]
            if isinstance(path, (str, bytes, os.PathLike)):
                resolved = str(Path(os.fsdecode(path)).resolve())
                if resolved.startswith(dataset_root + os.sep) or resolved == dataset_root:
                    raise RuntimeError('Dataset access is forbidden in serving worker')
    sys.addaudithook(deny_dataset)
    try:
        open(Path(dataset_root) / 'meta/info.json')
    except RuntimeError as exc:
        assert str(exc) == 'Dataset access is forbidden in serving worker'
    else:
        raise AssertionError('Serving dataset guard did not reject access')
    predictor = PolicyPredictor(checkpoint, args.device)
    observations = read_observations(args.observations)
    result = asyncio.run(exercise_websocket(checkpoint, predictor, observations,
                                            args.mode == 'temporal_aggregation',
                                            {'chunked': checkpoint['model_config']['num_queries'],
                                             'receding_horizon': 3}.get(args.mode, 1)))
    result['case'] = args.mode
    result['dataset_access'] = 'forbidden_by_audit_hook_in_fresh_process'
    save_json(args.result, result)


def run_variant(args, variant, observations):
    import torch
    from b1k_training import load_checkpoint, parser, train
    directory = args.output_dir / variant['name']
    directory.mkdir(parents=True, exist_ok=False)
    result = {'variant': variant, 'device': args.device, 'result': 'RUNNING', 'commands': []}
    result_path = directory / 'evidence.json'
    save_json(result_path, result)
    try:
        common = ['--dataset-path', str(args.dataset_path), '--task-names', *args.task_names,
                  '--output-dir', str(directory / 'training'), '--device', args.device, '--num-workers', '0',
                  '--batch-size', '2' if variant['policy'] == 'ACT' else '1', '--save-every', '1',
                  '--policy-class', variant['policy'], '--no-pretrained-backbone', '--chunk-size', '4',
                  '--hidden-dim', '32', '--dim-feedforward', '64', '--enc-layers', '1', '--dec-layers', '1',
                  '--nheads', '4', '--stats-max-frames', '64']
        if variant['name'] != 'cnnmlp-native':
            common += ['--image-size', *map(str, variant['image_size'])]
        if variant['policy'] == 'ACT':
            common += ['--position-embedding', variant['position'], '--pre-norm' if variant['pre_norm'] else '--no-pre-norm']
        first_args = common + ['--max-steps', '2']
        result['commands'].append([sys.executable, 'scripts/b1k/train_b1k.py', *first_args])
        step2 = train(parser().parse_args(first_args))
        before = load_checkpoint(step2)
        head = 'model.action_head.weight' if variant['policy'] == 'ACT' else 'model.mlp.4.weight'
        before_weight = before['model'][head].clone()
        before_optimizer_step = max(float(state['step']) for state in before['optimizer']['state'].values())
        assert before['step'] == before_optimizer_step == 2
        result['model_config'] = before['model_config']
        result['adapter_config'] = before['adapter_config']
        result['normalization'] = {key: before['normalization'][key] for key in ['count', 'approximate', 'fingerprint']}
        del before
        gc.collect()
        resumed_args = common + ['--max-steps', '3', '--resume', str(step2)]
        result['commands'].append([sys.executable, 'scripts/b1k/train_b1k.py', *resumed_args])
        step3 = train(parser().parse_args(resumed_args))
        restored = load_checkpoint(step3)
        optimizer_step = max(float(state['step']) for state in restored['optimizer']['state'].values())
        assert restored['step'] == optimizer_step == 3
        assert not torch.equal(before_weight, restored['model'][head])
        assert all(torch.isfinite(value).all() for value in restored['model'].values() if value.is_floating_point())
        result['resume'] = {'step2': str(step2), 'step3': str(step3), 'optimizer_step': optimizer_step,
                            'head_updated_after_resume': True, 'finite_parameters': True}
        result['metrics'] = [json.loads(line) for line in (directory / 'training/metrics.jsonl').read_text().splitlines()]
        assert [row['step'] for row in result['metrics']] == [1, 2, 3]
        digest = hashlib.sha256()
        with step3.open('rb') as file:
            for block in iter(lambda: file.read(1024 * 1024), b''):
                digest.update(block)
        result['checkpoint_sha256'] = digest.hexdigest()
        del restored, before_weight
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        modes = ['chunked', 'receding_horizon', 'temporal_aggregation'] if variant['policy'] == 'ACT' else ['single_action']
        result['serving'] = []
        for mode in modes:
            destination = directory / f'{mode}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--serve-worker', '--device', args.device,
                       '--checkpoint', str(step3), '--observations', str(observations), '--mode', mode,
                       '--result', str(destination), '--threads', str(args.threads)]
            result['commands'].append(command)
            with (directory / f'{mode}.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
            result['serving'].append(json.loads(destination.read_text()))
        result['result'] = 'PASS'
    except Exception:
        result['result'] = 'FAIL'
        result['error'] = traceback.format_exc()
        raise
    finally:
        save_json(result_path, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--dataset-path', type=Path, default=Path('/tmp/dev/datasets/2026-challenge-demos'))
    p.add_argument('--task-names', nargs='+', default=['turning_on_radio', 'picking_up_trash'])
    p.add_argument('--output-dir', type=Path, default=Path('outputs/variants'))
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--variants', nargs='+', choices=[v['name'] for v in MATRIX])
    p.add_argument('--serve-worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--checkpoint', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--observations', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--mode', choices=['chunked', 'receding_horizon', 'temporal_aggregation', 'single_action'],
                   help=argparse.SUPPRESS)
    p.add_argument('--result', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    # The parent checks CUDA idleness before importing torch; workers share its owned GPU.
    if args.serve_worker:
        if args.device == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        serving_worker(args)
        return
    device_environment(args.device)
    import torch
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError('Matrix output directory must be empty; choose a new directory')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [v for v in MATRIX if not args.variants or v['name'] in args.variants]
    report = {'device': args.device, 'torch': torch.__version__, 'result': 'RUNNING', 'matrix': selected,
              'gpu_validation': 'pending_idle_gpu' if args.device == 'cpu' else 'idle_checked_before_torch_import',
              'command': [sys.executable, *sys.argv], 'variants': []}
    started = time.monotonic()
    summary = args.output_dir / 'summary.json'
    try:
        observations = args.output_dir / 'real_observations.npz'
        report['observation_sources'] = save_observations(args.dataset_path, args.task_names, observations)
        for variant in selected:
            report['variants'].append(run_variant(args, variant, observations))
            save_json(summary, report)
        report['result'] = 'PASS'
    except Exception:
        report['result'] = 'FAIL'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_s'] = time.monotonic() - started
        save_json(summary, report)
    print(json.dumps({'result': report['result'], 'variants': len(report['variants']),
                      'serving_cases': sum(len(v['serving']) for v in report['variants']), 'summary': str(summary)}))


if __name__ == '__main__':
    main()
