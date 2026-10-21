#!/usr/bin/env python3
"""Exercise BEHAVIOR serving using a real local observation and independent sessions."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import websockets.asyncio.client
from websockets.exceptions import ConnectionClosedError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from b1k_dataset import B1KDataset, CAMERAS, GOAL_OBS_KEYS, OBS_KEYS
from b1k_server import packb, unpackb


async def exercise(args, observation, later):
    """`later` is the next frame of the same episode (another state), for the plan check."""
    url = f'ws://{args.host}:{args.port}'
    with urllib.request.urlopen(f'http://{args.host}:{args.port}/healthz') as response:
        assert response.status == 200 and response.read() == b'OK\n'
    async with websockets.asyncio.client.connect(url) as a, websockets.asyncio.client.connect(url) as b:
        metadata = unpackb(await a.recv())
        assert metadata == unpackb(await b.recv())
        assert metadata.get('served_actions', 'absolute') == 'absolute'
        horizon = metadata['action_horizon']
        async def request(client, obs):
            await client.send(packb(obs))
            action = unpackb(await client.recv())['action']
            batch = np.asarray(obs['robot_r1::proprio']).shape[0]
            assert action.shape == (batch, 23) and action.dtype == np.float32 and np.isfinite(action).all()
            return action
        first_a, first_b = await asyncio.gather(request(a, observation), request(b, observation))
        np.testing.assert_array_equal(first_a, first_b)
        await a.send(packb({'reset': True}))
        try:
            await asyncio.wait_for(a.recv(), .1)
            raise AssertionError('Reset unexpectedly sent a reply')
        except asyncio.TimeoutError:
            pass
        np.testing.assert_array_equal(await request(a, observation), first_a)
        if metadata.get('temporal_agg', False):
            for _ in range(metadata['chunk_size'] + 1):
                np.testing.assert_array_equal(await request(a, observation), await request(b, observation))
        else:
            for _ in range(horizon - 1):
                np.testing.assert_array_equal(await request(a, observation), await request(b, observation))
            np.testing.assert_array_equal(await request(a, observation), first_a)
        batch_obs = {key: np.repeat(value, 2, axis=0) if isinstance(value, np.ndarray) else value
                     for key, value in observation.items()}
        await request(a, batch_obs)
        tasks = [int(task) for task in metadata['task_map']]
        if len(tasks) > 1:
            batch_obs['task_id'] = np.array(tasks[:2], dtype=np.int64)
            await request(a, batch_obs)
            batch_obs['task_id'] = np.array(tasks[:2][::-1], dtype=np.int64)
            await request(a, batch_obs)
        await request(a, observation)
        invalid = dict(observation, task_id=999999)
        await a.send(packb(invalid))
        try:
            await a.recv()
            raise AssertionError('Unseen task accepted')
        except ConnectionClosedError as exc:
            assert exc.rcvd.code == 1008
    checks = ['health', 'handshake', 'float32_batch23', 'connection_isolation', 'reset_no_ack', 'replan', 'batch_resize',
              'task_change', 'unseen_task_rejected']
    if not metadata.get('temporal_agg') and horizon > 1:
        # a plan's actions are decoded (relative ones: plus the state) when it is made: the second action of the plan
        # made at `observation` does not depend on the observation that requests it
        async with websockets.asyncio.client.connect(url) as a, websockets.asyncio.client.connect(url) as b:
            await a.recv(), await b.recv()
            await asyncio.gather(request(a, observation), request(b, observation))
            np.testing.assert_array_equal(await request(a, later), await request(b, observation))
        checks.append('plan_decoded_at_plan_time')
    print(json.dumps({'result': 'PASS', 'metadata': metadata, 'checks': checks}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-path', '--dataset-root', dest='dataset_path', required=True)
    p.add_argument('--task-names', nargs='+', default=['turning_on_radio'])
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--goal-views', nargs='*', choices=CAMERAS, default=[CAMERAS[0]],
                   help='Cameras whose goal image (the episode\'s last recorded frame) each request carries; none for a '
                        'goal-free checkpoint or a server started with --goal-image')
    args = p.parse_args()
    dataset = B1KDataset(args.dataset_path, args.task_names, chunk_size=4)
    episode = dataset.episodes[1]['episode_index']
    # goal-conditioned servers get the goal views' last recorded frame of the episode (goal_source episode_last)
    goal_images = dataset.raw_sample(episode, dataset.by_id[episode]['length'] - 1)[0]  # every frame kept here
    observations = []
    for frame in (0, 1):
        images, state, _, _, task = dataset.raw_sample(episode, frame)
        observations.append({'robot_r1::proprio': state[None], 'task_id': task,
                             **{key: image[None] for key, image in zip(OBS_KEYS, images)},
                             **{GOAL_OBS_KEYS[camera]: goal_images[c][None] for c, camera in enumerate(CAMERAS)
                                if camera in args.goal_views}})
    dataset.close()
    asyncio.run(exercise(args, *observations))


if __name__ == '__main__':
    main()
