"""BEHAVIOR NumPy MessagePack protocol with per-connection ACT/CNNMLP execution."""

import argparse
import asyncio
import http
import logging

import msgpack
import numpy as np
import torch
import websockets.asyncio.server as websocket_server
from websockets.exceptions import ConnectionClosed

from b1k_dataset import CAMERAS, GOAL_OBS_KEYS, OBS_KEYS, preprocess_image, preprocess_state
from b1k_training import goal_config, load_checkpoint, make_policy, policy_class
from b1k_language import language_embedding_table, language_for_tasks


LOGGER = logging.getLogger(__name__)


def load_goal_images(specs):
    """`--goal-image CAMERA=PATH` values to {camera: uint8 HWC RGB array} (PNG/JPEG via PyAV, no extra dependency)."""
    import av
    goals = {}
    for spec in specs or []:
        camera, _, path = spec.partition('=')
        if camera not in CAMERAS or not path:
            raise ValueError(f'--goal-image expects CAMERA=PATH with CAMERA in {CAMERAS}, got {spec!r}')
        with av.open(path) as container:
            frame = next(container.decode(video=0))
            goals[camera] = frame.to_ndarray(format='rgb24')
    return goals


def pack_array(value):
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ('V', 'O', 'c'):
        raise ValueError(f'Unsupported dtype {value.dtype}')
    if isinstance(value, np.ndarray):
        return {b'__ndarray__': True, b'data': value.tobytes(), b'dtype': value.dtype.str, b'shape': value.shape}
    if isinstance(value, np.generic):
        return {b'__npgeneric__': True, b'data': value.item(), b'dtype': value.dtype.str}
    raise TypeError(f'Cannot encode {type(value)}')


def unpack_array(value):
    if b'__ndarray__' in value or b'__npgeneric__' in value:
        dtype = np.dtype(value[b'dtype'])
        if dtype.kind in ('V', 'O', 'c'):
            raise ValueError(f'Unsupported dtype {dtype}')
        if b'__ndarray__' in value:
            return np.frombuffer(value[b'data'], dtype=dtype).reshape(value[b'shape'])
        return dtype.type(value[b'data'])
    return value


def packb(value):
    return msgpack.packb(value, default=pack_array, use_bin_type=True)


def unpackb(value):
    return msgpack.unpackb(value, object_hook=unpack_array, raw=False, strict_map_key=False)


class PolicyPredictor:
    def __init__(self, checkpoint, device='cuda'):
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        embeddings = language_embedding_table(checkpoint['model_config'], checkpoint['task_map'],
                                               checkpoint.get('language_cache'))
        self.language_embeddings = embeddings.to(self.device) if embeddings is not None else None
        self.task_map = checkpoint['task_map']
        self.goal_views = goal_config(checkpoint['model_config'])['goal_views']
        self.policy = make_policy(checkpoint['model_config'], self.device, restoring=True)
        self.policy.load_state_dict(checkpoint['model'])
        self.policy.eval()

    @torch.inference_mode()
    def __call__(self, qpos, images, task_ids=None, goal=None):
        """Denormalized actions for a batch; `task_ids` (B,) select the cached prompt, `goal` is (B, views, 3, H, W)."""
        qpos, images = qpos.to(self.device), images.to(self.device)
        kwargs = {}
        if self.language_embeddings is not None:
            if task_ids is None:
                raise ValueError('Language-conditioned checkpoints need task ids to select the prompt')
            kwargs['lang_emb'] = language_for_tasks(torch.as_tensor(np.asarray(task_ids), device=self.device),
                                                    self.language_embeddings, self.task_map)
        if self.goal_views:
            if goal is None:
                raise ValueError(f'Goal-conditioned checkpoint needs goal images for views {self.goal_views}')
            kwargs['goal'] = goal.to(self.device)
        actions = self.policy(qpos, images, **kwargs).cpu().numpy()
        stats = self.checkpoint['normalization']
        return (actions * np.asarray(stats['action_std'], dtype=np.float32) +
                np.asarray(stats['action_mean'], dtype=np.float32)).astype(np.float32)


# Compatibility for clients importing the original B1K predictor name.
ACTPredictor = PolicyPredictor


class Session:
    def __init__(self, predictor, checkpoint, action_horizon=None, task_name=None, temporal_agg=False,
                 fixed_goals=None):
        self.predictor = predictor
        self.task_map = checkpoint['task_map']
        self.stats = checkpoint['normalization']
        adapter = checkpoint['adapter_config']
        self.image_size = adapter['image_size']
        self.task_onehot = adapter.get('task_conditioning', 'onehot') == 'onehot'
        # Goal images: per request under the `goal::<camera observation key>` keys, else the fixed images given at
        # server start (--goal-image CAMERA=PATH; an externally supplied target image), else the request is rejected.
        self.goal_views = list(adapter.get('goal_views', []))
        self.goal_keys = {view: GOAL_OBS_KEYS[view] for view in self.goal_views}
        self.fixed_goals = {view: np.asarray(image) for view, image in (fixed_goals or {}).items() if view in self.goal_views}
        self.policy_class = policy_class(checkpoint['model_config'])
        self.chunk_size = checkpoint['model_config']['num_queries']
        self.temporal_agg = temporal_agg
        if self.policy_class == 'CNNMLP' and temporal_agg:
            raise ValueError('Temporal aggregation is only supported for ACT; CNNMLP predicts one action')
        if action_horizon is None:
            action_horizon = 1 if temporal_agg or self.policy_class == 'CNNMLP' else min(16, self.chunk_size)
        if not 1 <= action_horizon <= self.chunk_size:
            raise ValueError(f'--action-horizon must be between 1 and checkpoint chunk size {self.chunk_size}')
        if temporal_agg and action_horizon != 1:
            raise ValueError('ACT temporal aggregation queries every observation; --action-horizon must be 1')
        self.action_horizon = action_horizon
        self.default_task = next(iter(self.task_map)) if len(self.task_map) == 1 else None
        if task_name is not None:
            matching = [i for i, name in self.task_map.items() if name == task_name]
            if not matching:
                raise ValueError(f'Unseen task name {task_name}; checkpoint tasks: {self.task_map}')
            self.default_task = matching[0]
        self.reset()

    def reset(self):
        self.plans = None
        self.positions = None
        self.task_ids = None
        self.history = []

    def resolve_tasks(self, value, batch_size):
        if value is None:
            if self.default_task is None:
                raise ValueError('Multi-task checkpoint requires task_id in observations or --task-name')
            value = [self.default_task]
        values = np.asarray(value).reshape(-1)
        if values.size not in (1, batch_size) or values.dtype.kind not in 'iu':
            raise ValueError('task_id must contain one integer or one integer per batch element')
        values = np.broadcast_to(values, (batch_size,)).copy()
        unknown = set(values.tolist()) - set(self.task_map)
        if unknown:
            raise ValueError(f'Unseen task_id {sorted(unknown)}; checkpoint tasks: {self.task_map}')
        return values

    def act(self, observation):
        if not isinstance(observation, dict):
            raise ValueError('Observation must be a map')
        state = np.asarray(observation['robot_r1::proprio'])
        if state.ndim == 1:
            state = state[None]
        if state.ndim != 2 or state.shape[1] != 61 or state.shape[0] < 1:
            raise ValueError('robot_r1::proprio must have shape (61,) or (B,61)')
        batch_size = len(state)
        task_ids = self.resolve_tasks(observation.get('task_id'), batch_size)
        qpos = torch.stack([preprocess_state(s, int(task), self.stats, self.task_map, self.task_onehot)
                            for s, task in zip(state, task_ids)])
        images = self.batch_images(observation, OBS_KEYS, batch_size)
        goal = None
        if self.goal_views:
            goal_batch = {}
            for view, key in self.goal_keys.items():
                if key in observation:
                    goal_batch[key] = observation[key]
                elif view in self.fixed_goals:
                    goal_batch[key] = np.broadcast_to(self.fixed_goals[view], (batch_size, *self.fixed_goals[view].shape))
                else:
                    raise ValueError(f'Goal-conditioned checkpoint needs {key} in the observation (or --goal-image {view}=PATH)')
            goal = self.batch_images(goal_batch, list(self.goal_keys.values()), batch_size)
        if self.policy_class == 'CNNMLP':
            predictions = self.predictor(qpos, images, task_ids)
            self.validate_predictions(predictions, (batch_size, 23))
            self.task_ids = task_ids
            return predictions.astype(np.float32)
        if self.temporal_agg:
            return self.aggregate(qpos, images, task_ids, goal)
        if self.plans is None or len(self.plans) != batch_size:
            plans = np.zeros((batch_size, self.action_horizon, 23), dtype=np.float32)
            positions = np.full(batch_size, self.action_horizon, dtype=np.int64)
        else:
            plans = self.plans.copy()
            positions = self.positions.copy()
            positions[task_ids != self.task_ids] = self.action_horizon
        needs_plan = positions >= self.action_horizon
        if needs_plan.any():
            predictions = self.predictor(qpos[needs_plan], images[needs_plan], task_ids[needs_plan],
                                         None if goal is None else goal[torch.from_numpy(needs_plan)])
            expected = (int(needs_plan.sum()), self.chunk_size, 23)
            self.validate_predictions(predictions, expected)
            plans[needs_plan] = predictions[:, :self.action_horizon]
            positions[needs_plan] = 0
        action = plans[np.arange(batch_size), positions].copy()
        self.plans, self.positions, self.task_ids = plans, positions + 1, task_ids
        return action.astype(np.float32)

    def batch_images(self, observation, keys, batch_size):
        """uint8 (B,H,W,3|4) arrays under `keys` to one float (B, len(keys), 3, H, W) tensor at the training size."""
        cameras = []
        for key in keys:
            camera = np.asarray(observation[key])
            if camera.ndim == 3:
                camera = camera[None]
            if camera.ndim != 4 or camera.shape[0] != batch_size:
                raise ValueError(f'{key} must have shape (B,H,W,3|4) with B={batch_size}')
            cameras.append(torch.stack([preprocess_image(image, self.image_size) for image in camera]))
        return torch.stack(cameras, dim=1)

    @staticmethod
    def validate_predictions(predictions, expected):
        if predictions.shape != expected or not np.isfinite(predictions).all():
            raise ValueError(f'Invalid model output; expected finite {expected}, got {predictions.shape}')

    def aggregate(self, qpos, images, task_ids, goal=None):
        predictions = self.predictor(qpos, images, task_ids, goal)
        batch_size = len(task_ids)
        self.validate_predictions(predictions, (batch_size, self.chunk_size, 23))
        history = []
        if self.task_ids is not None and len(self.task_ids) == batch_size:
            unchanged = task_ids == self.task_ids
            for remaining, valid in self.history:
                history.append((remaining, valid & unchanged))
        history.append((predictions, np.ones(batch_size, dtype=bool)))
        action = np.empty((batch_size, 23), dtype=np.float32)
        for slot in range(batch_size):
            # Preserve upstream oldest-first weights; validity does not depend on action values.
            candidates = np.stack([remaining[slot, 0] for remaining, valid in history if valid[slot]])
            weights = np.exp(-0.01 * np.arange(len(candidates)))
            weights /= weights.sum()
            action[slot] = (candidates * weights[:, None]).sum(axis=0)
        self.history = [(remaining[:, 1:].copy(), valid) for remaining, valid in history
                        if remaining.shape[1] > 1 and valid.any()]
        self.task_ids = task_ids
        return action


class B1KServer:
    def __init__(self, checkpoint, predictor, host='0.0.0.0', port=8000, action_horizon=None,
                 task_name=None, temporal_agg=False, fixed_goals=None):
        self.checkpoint, self.predictor = checkpoint, predictor
        self.host, self.port = host, port
        self.fixed_goals = fixed_goals or {}
        session = Session(predictor, checkpoint, action_horizon, task_name, temporal_agg, self.fixed_goals)
        self.action_horizon, self.task_name = session.action_horizon, task_name
        self.temporal_agg = temporal_agg
        mode = 'single_action' if session.policy_class == 'CNNMLP' else ('temporal_aggregation' if temporal_agg else 'chunked')
        goal = goal_config(checkpoint['model_config'])
        self.metadata = {'policy': session.policy_class, 'protocol': 'behavior-numpy-msgpack', 'action_dim': 23,
                         'action_horizon': self.action_horizon, 'chunk_size': session.chunk_size,
                         'execution_mode': mode, 'temporal_agg': temporal_agg,
                         'temporal_agg_decay': 0.01 if temporal_agg else None,
                         'task_map': {str(k): v for k, v in checkpoint['task_map'].items()}, 'proprio_dim': 61,
                         'observation_keys': OBS_KEYS, 'checkpoint_step': checkpoint['step'],
                         'language_conditioning': checkpoint['model_config'].get('language_conditioning', 'none'),
                         'language_encoder': (checkpoint['model_config'].get('language_encoder', 'clip')
                                              if checkpoint['model_config'].get('language_conditioning', 'none') != 'none'
                                              else None),
                         'prompt_source': checkpoint['model_config'].get('prompt_source', 'task_name'),
                         'regime': checkpoint['model_config'].get('regime'),
                         'task_conditioning': checkpoint['adapter_config'].get('task_conditioning', 'onehot'),
                         'goal_fusion': goal['goal_fusion'], 'goal_views': goal['goal_views'],
                         'goal_observation_keys': [GOAL_OBS_KEYS[view] for view in goal['goal_views']],
                         'goal_source': checkpoint['adapter_config'].get('goal_source') if goal['goal_views'] else None,
                         'fixed_goal_views': sorted(self.fixed_goals)}

    async def handler(self, websocket):
        session = Session(self.predictor, self.checkpoint, self.action_horizon, self.task_name, self.temporal_agg,
                          self.fixed_goals)
        await websocket.send(packb(self.metadata))
        try:
            async for message in websocket:
                try:
                    observation = unpackb(message)
                    if isinstance(observation, dict) and 'reset' in observation:
                        session.reset()
                        continue
                    action = session.act(observation)
                    await websocket.send(packb({'action': action}))
                except (ValueError, KeyError, TypeError, msgpack.ExtraData) as exc:
                    LOGGER.warning('Rejected observation: %s', exc)
                    await websocket.close(code=1008, reason=str(exc)[:100])
                    return
        except ConnectionClosed:
            pass
        except Exception:
            LOGGER.exception('Inference failed')
            await websocket.close(code=1011, reason='Policy inference failed')
        finally:
            session.reset()

    @staticmethod
    def health(connection, request):
        if request.path == '/healthz':
            return connection.respond(http.HTTPStatus.OK, 'OK\n')
        return None

    async def run(self):
        async with websocket_server.serve(self.handler, self.host, self.port, compression=None,
                                          max_size=64 * 1024 * 1024, process_request=self.health) as server:
            LOGGER.info('%s B1K ready at ws://%s:%d; HTTP /healthz', self.metadata['policy'], self.host, self.port)
            await server.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--action-horizon', type=int,
                   help='Cached ACT actions (default min(16, chunk size)); CNNMLP/aggregation require 1')
    p.add_argument('--temporal-agg', action='store_true', help='ACT only: query each step and aggregate overlapping chunks')
    p.add_argument('--task-name', help='Default task if task_id is absent; must exist in checkpoint')
    p.add_argument('--goal-image', action='append', metavar='CAMERA=PATH',
                   help='Goal-conditioned checkpoints: fixed goal image (PNG/JPEG, RGB) for a camera view, used when a '
                        'request carries no goal::<camera key> image; repeat per view')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    checkpoint = load_checkpoint(args.model_path)
    server = B1KServer(checkpoint, PolicyPredictor(checkpoint, args.device), args.host, args.port,
                       args.action_horizon, args.task_name, args.temporal_agg, load_goal_images(args.goal_image))
    asyncio.run(server.run())
