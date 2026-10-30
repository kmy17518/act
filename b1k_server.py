"""BEHAVIOR NumPy MessagePack protocol with per-connection ACT/CNNMLP execution."""

import argparse
import asyncio
import http
import json
import logging

import msgpack
import numpy as np
import torch
import websockets.asyncio.server as websocket_server
from websockets.exceptions import ConnectionClosed

from b1k_dataset import (ACTION_DELTA, CAMERAS, GOAL_OBS_KEYS, OBS_KEYS, action_reference, decode_actions, extract_state,
                         preprocess_image, preprocess_state, split_stop_signals)
from b1k_training import goal_config, load_checkpoint, make_policy, policy_class
from b1k_language import language_embedding_table, language_for_tasks


LOGGER = logging.getLogger(__name__)
# Stop study, Method A gating defaults (b1k_server --stop-threshold ... ; `StopGate`): DONE when the stop signal is at
# least STOP_THRESHOLD and the progress signal at least PROGRESS_THRESHOLD for STOP_HOLD_STEPS consecutive control
# steps (30 = 1 s at 30 Hz).
STOP_THRESHOLD = 0.5
PROGRESS_THRESHOLD = 0.9
STOP_HOLD_STEPS = 30


class StopGate:
    """Per-slot DONE decision from a --stop-dims checkpoint's signals, and the pose held once DONE.

    `update(stop, progress, state, action)` counts consecutive control steps with stop >= threshold and
    progress >= progress_threshold; at `hold_steps` of them the slot is DONE (latched until `reset`), and the hold
    pose is fixed: the torso / arm joints at the state of that step, zero base velocity, the grippers at the last
    executed command (a command derived from the finger opening could differ from the one holding an object).
    `enabled=False` keeps counting and reporting but never declares DONE (signals-only serving)."""
    def __init__(self, threshold=STOP_THRESHOLD, progress_threshold=PROGRESS_THRESHOLD, hold_steps=STOP_HOLD_STEPS,
                 enabled=True):
        if not 0 <= threshold <= 1 or not 0 <= progress_threshold <= 1 or hold_steps < 1:
            raise ValueError('stop / progress thresholds must be in [0, 1] and hold_steps positive')
        self.threshold, self.progress_threshold, self.hold_steps, self.enabled = threshold, progress_threshold, hold_steps, enabled
        self.reset()

    def reset(self):
        self.count = None
        self.done = None
        self.hold = None

    def config(self):
        return {'stop_threshold': self.threshold, 'progress_threshold': self.progress_threshold,
                'hold_steps': self.hold_steps, 'enabled': self.enabled, 'done_action': 'hold_pose'}

    def update(self, stop, progress, state, action, keep=None):
        """`stop`, `progress` (B,) signals of this control step, `state` (B, 61) raw states, `action` (B, 23) the
        absolute action that would be executed; `keep` (B,) False resets a slot (task change). Returns
        (action to execute (B, 23), done (B,))."""
        batch = len(stop)
        if self.count is None or len(self.count) != batch:
            self.count = np.zeros(batch, dtype=np.int64)
            self.done = np.zeros(batch, dtype=bool)
            self.hold = np.zeros((batch, action.shape[-1]), dtype=np.float32)
        if keep is not None:
            self.count[~keep] = 0
            self.done[~keep] = False
        active = (stop >= self.threshold) & (progress >= self.progress_threshold)
        self.count = np.where(active, self.count + 1, 0)
        newly = self.enabled & (self.count >= self.hold_steps) & ~self.done
        if newly.any():
            hold = action_reference(extract_state(np.asarray(state, dtype=np.float32)[newly], 'sum'), ACTION_DELTA)
            hold[:, ACTION_DELTA['gripper_action_dims']] = action[newly][:, ACTION_DELTA['gripper_action_dims']]
            self.hold[newly] = hold.astype(np.float32)
            self.done[newly] = True
        action = np.where(self.done[:, None], self.hold, action).astype(np.float32)
        return action, self.done.copy()


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
    def __init__(self, checkpoint, device='cuda', jpeg_quality=None, jpeg_subsampling=0):
        self.jpeg_quality, self.jpeg_subsampling = jpeg_quality, jpeg_subsampling
        if jpeg_quality is not None and (not 1 <= jpeg_quality <= 100 or jpeg_subsampling not in (0,1,2)):
            raise ValueError('invalid JPEG quality/subsampling')
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
    def predict(self, qpos, images, task_ids=None, goal=None, states=None):
        """(absolute actions in action units (B, T, 23), signals) for a batch; `task_ids` (B,) select the cached
        prompt, `goal` is (B, views, 3, H, W). Relative-action checkpoints need `states`, the observations' raw
        (B, 61) R1Pro states the prediction is made from: their chunks are relative to them. `signals` are the
        stop / progress columns (B, T, 2) in [0, 1] of a --stop-dims checkpoint, else None."""
        if self.jpeg_quality is not None:
            from b1k_cache_v2 import jpeg_roundtrip_chw
            images = jpeg_roundtrip_chw(images, self.jpeg_quality, self.jpeg_subsampling)
            if goal is not None:
                goal = jpeg_roundtrip_chw(goal, self.jpeg_quality, self.jpeg_subsampling)
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
        outputs = self.policy(qpos, images, **kwargs).cpu().numpy()
        actions, signals = split_stop_signals(outputs)
        proprio = None if states is None else extract_state(np.asarray(states, dtype=np.float32), 'sum')
        return decode_actions(actions, self.checkpoint['normalization'], proprio).astype(np.float32), signals

    def __call__(self, qpos, images, task_ids=None, goal=None, states=None):
        """The absolute actions of `predict` (the signals of a --stop-dims checkpoint dropped)."""
        return self.predict(qpos, images, task_ids, goal, states)[0]


# Compatibility for clients importing the original B1K predictor name.
ACTPredictor = PolicyPredictor

GRIPPER_ACTION_DIMS = [14, 22]  # left and right gripper commands, in [-1, 1] (+1 = fully open)
DEFAULT_GRIPPER_SNAP = 0.9


def snap_grippers(actions, threshold):
    """Gripper commands clamped to [-1, 1], with those above `threshold` set to exactly +1 (None: untouched).

    OmniGibson releases an assisted grasp only when the finger target reaches the open limit, i.e. at a command of
    exactly +1, while a regressed (or temporally averaged) opening ramp tends to end just below it. In the training
    tasks no gripper hold sits at or above 0.80, so commands above 0.9 belong to opening ramps.
    """
    if threshold is None:
        return actions
    actions = actions.copy()
    grippers = np.clip(actions[..., GRIPPER_ACTION_DIMS], -1.0, 1.0)
    grippers[grippers > threshold] = 1.0
    actions[..., GRIPPER_ACTION_DIMS] = grippers
    return actions


class Session:
    """One client's execution state: cached chunks (or the temporal-aggregation history), and for a --stop-dims
    checkpoint the stop / progress signals of every control step (`signals` after `act`) with the `StopGate` that
    turns them into DONE: chunked execution summarizes a new chunk's first `stop_positions` steps (default: the
    action horizon) by their mean and keeps that until the next chunk; temporal aggregation uses the aggregated
    current-step values (the signal columns are averaged over the overlapping chunks like the actions)."""
    def __init__(self, predictor, checkpoint, action_horizon=None, task_name=None, temporal_agg=False,
                 fixed_goals=None, gripper_snap=None, stop_gate=None, stop_positions=None):
        self.predictor = predictor
        self.gripper_snap = gripper_snap  # applied to every returned action (`snap_grippers`); B1KServer sets 0.9
        self.stop_signals = bool(checkpoint['model_config'].get('stop_dims', False))
        self.stop_gate = stop_gate if stop_gate is not None else StopGate()
        self.signals = None
        self.task_map = checkpoint['task_map']
        self.stats = checkpoint['normalization']
        adapter = checkpoint['adapter_config']
        self.image_size = adapter['image_size']
        self.task_onehot = adapter.get('task_conditioning', 'onehot') == 'onehot'
        self.gripper_state = adapter.get('gripper_state', 'fingers')
        # Goal images: per request under the `goal::<camera observation key>` keys, else the fixed images given at
        # server start (--goal-image CAMERA=PATH; an externally supplied target image), else the request is rejected.
        self.goal_views = list(adapter.get('goal_views', []))
        self.goal_keys = {view: GOAL_OBS_KEYS[view] for view in self.goal_views}
        self.fixed_goals = {view: np.asarray(image) for view, image in (fixed_goals or {}).items() if view in self.goal_views}
        self.policy_class = policy_class(checkpoint['model_config'])
        self.chunk_size = checkpoint['model_config']['num_queries']
        # relative chunks become absolute with the state of their own observation (the predictor's `states`), before
        # a plan is stored or overlapping chunks are aggregated, and before the gripper snap
        self.relative = checkpoint['model_config'].get('action_representation', 'absolute') == 'relative'
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
        if stop_positions is None:
            stop_positions = 1 if temporal_agg else action_horizon
        if not 1 <= stop_positions <= self.chunk_size or (temporal_agg and stop_positions != 1):
            raise ValueError(f'--stop-positions must be between 1 and the chunk size {self.chunk_size} (1 with temporal '
                             f'aggregation, which averages the current step over the overlapping chunks)')
        self.stop_positions = stop_positions
        self.default_task = next(iter(self.task_map)) if len(self.task_map) == 1 else None
        if task_name is not None:
            matching = [i for i, name in self.task_map.items() if name == task_name]
            if not matching:
                raise ValueError(f'Unseen task name {task_name}; checkpoint tasks: {self.task_map}')
            self.default_task = matching[0]
        self.reset()

    def reset(self):
        self.plans = None
        self.summaries = None
        self.positions = None
        self.task_ids = None
        self.history = []
        self.signals = None
        self.stop_gate.reset()

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
        qpos = torch.stack([preprocess_state(s, int(task), self.stats, self.task_map, self.task_onehot,
                                             gripper_state=self.gripper_state)
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
            predictions, _ = self.predict(state, qpos, images, task_ids)
            self.validate_predictions(predictions, (batch_size, 23))
            self.task_ids = task_ids
            return snap_grippers(predictions.astype(np.float32), self.gripper_snap)
        keep = None
        if self.task_ids is not None and len(self.task_ids) == batch_size:
            keep = task_ids == self.task_ids
        if self.temporal_agg:
            # the weighted average of overlapping chunks can bring a +1 opening below 1, so snap after it
            action, signals = self.aggregate(qpos, images, task_ids, goal, state)
        else:
            action, signals = self.chunked(qpos, images, task_ids, goal, state, batch_size)
        action = snap_grippers(action.astype(np.float32), self.gripper_snap)
        self.signals = None
        if signals is not None:
            action, done = self.stop_gate.update(signals[:, 0], signals[:, 1], state, action, keep)
            self.signals = {'stop': signals[:, 0].astype(np.float32), 'progress': signals[:, 1].astype(np.float32),
                            'done': done, 'hold_count': self.stop_gate.count.copy()}
        return action

    def chunked(self, qpos, images, task_ids, goal, state, batch_size):
        """One action per slot from its cached chunk (a new chunk when it is used up or the task changed), plus the
        chunk's signal summary (B, 2) for a --stop-dims checkpoint (None otherwise)."""
        if self.plans is None or len(self.plans) != batch_size:
            plans = np.zeros((batch_size, self.action_horizon, 23), dtype=np.float32)
            summaries = np.zeros((batch_size, 2), dtype=np.float32)
            positions = np.full(batch_size, self.action_horizon, dtype=np.int64)
        else:
            plans, summaries = self.plans.copy(), self.summaries.copy()
            positions = self.positions.copy()
            positions[task_ids != self.task_ids] = self.action_horizon
        needs_plan = positions >= self.action_horizon
        if needs_plan.any():
            predictions, signals = self.predict(state[needs_plan], qpos[needs_plan], images[needs_plan],
                                                task_ids[needs_plan], None if goal is None else goal[torch.from_numpy(needs_plan)])
            count = int(needs_plan.sum())
            self.validate_predictions(predictions, (count, self.chunk_size, 23))
            plans[needs_plan] = predictions[:, :self.action_horizon]
            positions[needs_plan] = 0
            if signals is not None:
                self.validate_predictions(signals, (count, self.chunk_size, 2))
                summaries[needs_plan] = signals[:, :self.stop_positions].mean(axis=1)
        action = plans[np.arange(batch_size), positions].copy()
        self.plans, self.summaries, self.positions, self.task_ids = plans, summaries, positions + 1, task_ids
        return action, (summaries.copy() if self.stop_signals else None)

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

    def predict(self, state, *args):
        """(absolute actions, signals or None) of the predictor for `args` (qpos, images, task_ids[, goal]); a
        relative checkpoint's predictor decodes them with the observations' raw `state`. Predictors without a
        `predict` method (plain callables returning actions) give no signals."""
        kwargs = {'states': state} if self.relative else {}
        if hasattr(self.predictor, 'predict'):
            return self.predictor.predict(*args, **kwargs)
        return self.predictor(*args, **kwargs), None

    def aggregate(self, qpos, images, task_ids, goal=None, state=None):
        """Temporal aggregation: (the weighted average of the overlapping chunks' current step (B, 23), the same
        average of the signal columns (B, 2) for a --stop-dims checkpoint, else None)."""
        predictions, signals = self.predict(state, qpos, images, task_ids, goal)
        batch_size = len(task_ids)
        self.validate_predictions(predictions, (batch_size, self.chunk_size, 23))
        if signals is not None:
            self.validate_predictions(signals, (batch_size, self.chunk_size, 2))
            predictions = np.concatenate([predictions, signals.astype(np.float32)], axis=-1)
        history = []
        if self.task_ids is not None and len(self.task_ids) == batch_size:
            unchanged = task_ids == self.task_ids
            for remaining, valid in self.history:
                history.append((remaining, valid & unchanged))
        history.append((predictions, np.ones(batch_size, dtype=bool)))
        action = np.empty((batch_size, predictions.shape[-1]), dtype=np.float32)
        for slot in range(batch_size):
            # Preserve upstream oldest-first weights; validity does not depend on action values.
            candidates = np.stack([remaining[slot, 0] for remaining, valid in history if valid[slot]])
            weights = np.exp(-0.01 * np.arange(len(candidates)))
            weights /= weights.sum()
            action[slot] = (candidates * weights[:, None]).sum(axis=0)
        self.history = [(remaining[:, 1:].copy(), valid) for remaining, valid in history
                        if remaining.shape[1] > 1 and valid.any()]
        self.task_ids = task_ids
        return action[:, :23], (action[:, 23:] if signals is not None else None)


class B1KServer:
    def __init__(self, checkpoint, predictor, host='0.0.0.0', port=8000, action_horizon=None,
                 task_name=None, temporal_agg=False, fixed_goals=None, gripper_snap=DEFAULT_GRIPPER_SNAP,
                 stop_gate=None, stop_positions=None, signal_log=None):
        self.checkpoint, self.predictor = checkpoint, predictor
        self.host, self.port = host, port
        self.fixed_goals = fixed_goals or {}
        self.gripper_snap = gripper_snap
        self.stop_gate_config = (stop_gate if stop_gate is not None else StopGate()).config()
        self.stop_positions = stop_positions
        self.signal_log = signal_log  # path of a JSON-lines log of every control step's signals (--stop-dims checkpoints)
        self.sessions = 0
        self.action_horizon, self.task_name, self.temporal_agg = action_horizon, task_name, temporal_agg
        session = self.session()  # validates the options; client sessions are numbered from 1
        self.sessions = 0
        self.action_horizon = session.action_horizon
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
                         'fixed_goal_views': sorted(self.fixed_goals),
                         'action_representation': checkpoint['model_config'].get('action_representation', 'absolute'),
                         'normalization': checkpoint['model_config'].get('normalization', 'zscore'),
                         'served_actions': 'absolute',
                         'gripper_action_dims': GRIPPER_ACTION_DIMS,
                         'gripper_clamp': [-1.0, 1.0] if gripper_snap is not None else None,
                         'gripper_snap_threshold': gripper_snap,
                         # --stop-dims checkpoints: every response also carries stop / progress (B,) float32 in
                         # [0, 1] and done (B,) bool; a DONE slot's action is its hold pose until the next reset
                         'stop_signals': ({'response_keys': ['stop', 'progress', 'done'], 'stop_positions': session.stop_positions,
                                           **self.stop_gate_config} if session.stop_signals else None)}

    def session(self):
        self.sessions += 1
        gate = StopGate(self.stop_gate_config['stop_threshold'], self.stop_gate_config['progress_threshold'],
                        self.stop_gate_config['hold_steps'], self.stop_gate_config['enabled'])
        return Session(self.predictor, self.checkpoint, self.action_horizon, self.task_name, self.temporal_agg,
                       self.fixed_goals, self.gripper_snap, gate, self.stop_positions)

    def log_signals(self, session_id, step, signals):
        if not self.signal_log or signals is None:
            return
        with open(self.signal_log, 'a') as handle:
            for slot in range(len(signals['stop'])):
                handle.write(json.dumps({'session': session_id, 'step': step, 'slot': slot,
                                         'stop': float(signals['stop'][slot]), 'progress': float(signals['progress'][slot]),
                                         'hold_count': int(signals['hold_count'][slot]), 'done': bool(signals['done'][slot])})
                             + '\n')

    async def handler(self, websocket):
        session = self.session()
        session_id, step = self.sessions, 0
        await websocket.send(packb(self.metadata))
        try:
            async for message in websocket:
                try:
                    observation = unpackb(message)
                    if isinstance(observation, dict) and 'reset' in observation:
                        session.reset()
                        self.sessions += 1
                        session_id, step = self.sessions, 0
                        continue
                    action = session.act(observation)
                    step += 1
                    payload = {'action': action}
                    if session.signals is not None:
                        payload.update(stop=session.signals['stop'], progress=session.signals['progress'],
                                       done=session.signals['done'])
                        self.log_signals(session_id, step, session.signals)
                    await websocket.send(packb(payload))
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
    p.add_argument('--jpeg-quality', type=int, help='Opt-in v2 serving parity after resize; match cache quality')
    p.add_argument('--jpeg-subsampling', type=int, choices=(0,1,2), default=0)
    p.add_argument('--action-horizon', type=int,
                   help='Cached ACT actions (default min(16, chunk size)); CNNMLP/aggregation require 1')
    p.add_argument('--temporal-agg', action='store_true', help='ACT only: query each step and aggregate overlapping chunks')
    p.add_argument('--task-name', help='Default task if task_id is absent; must exist in checkpoint')
    p.add_argument('--goal-image', action='append', metavar='CAMERA=PATH',
                   help='Goal-conditioned checkpoints: fixed goal image (PNG/JPEG, RGB) for a camera view, used when a '
                        'request carries no goal::<camera key> image; repeat per view')
    p.add_argument('--gripper-snap-threshold', type=lambda v: None if v.lower() == 'none' else float(v),
                   default=DEFAULT_GRIPPER_SNAP, metavar='T',
                   help='Clamp the gripper commands (action dims 14, 22) to [-1, 1] and send those above T fully open '
                        '(+1), the only command at which OmniGibson releases an assisted grasp (default 0.9); "none" '
                        'returns them as predicted')
    p.add_argument('--stop-threshold', type=float, default=STOP_THRESHOLD, metavar='T',
                   help='--stop-dims checkpoints: a control step counts towards DONE when its stop signal is at least T '
                        f'(default {STOP_THRESHOLD})')
    p.add_argument('--progress-threshold', type=float, default=PROGRESS_THRESHOLD, metavar='T',
                   help=f'... and its progress signal at least T (default {PROGRESS_THRESHOLD})')
    p.add_argument('--stop-hold-steps', type=int, default=STOP_HOLD_STEPS, metavar='N',
                   help=f'DONE after N consecutive such control steps (default {STOP_HOLD_STEPS}: 1 s at 30 Hz); the '
                        'slot then holds its pose (joints at that step\'s state, zero base velocity, the last gripper '
                        'commands) until the next reset')
    p.add_argument('--stop-positions', type=int, default=None, metavar='K',
                   help='Chunked execution: a new chunk\'s stop / progress signals are the means over its first K steps '
                        '(default: the action horizon); temporal aggregation averages the current step over the '
                        'overlapping chunks instead (K = 1)')
    p.add_argument('--no-stop-gating', action='store_true',
                   help='Serve and log the stop / progress signals but never declare DONE (the baseline termination '
                        'stays with the client)')
    p.add_argument('--signal-log', metavar='PATH',
                   help='Append every control step\'s stop / progress / hold count / done per slot as JSON lines')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    checkpoint = load_checkpoint(args.model_path)
    gate = StopGate(args.stop_threshold, args.progress_threshold, args.stop_hold_steps, not args.no_stop_gating)
    server = B1KServer(checkpoint, PolicyPredictor(checkpoint, args.device, args.jpeg_quality, args.jpeg_subsampling), args.host, args.port,
                       args.action_horizon, args.task_name, args.temporal_agg, load_goal_images(args.goal_image),
                       args.gripper_snap_threshold, gate, args.stop_positions, args.signal_log)
    if server.metadata['stop_signals']:
        LOGGER.info('Stop signals: %s', server.metadata['stop_signals'])
    asyncio.run(server.run())
