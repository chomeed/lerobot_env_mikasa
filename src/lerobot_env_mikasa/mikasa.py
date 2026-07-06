"""MIKASA-Robo-VLA environment wrapper for LeRobot evaluation.

Wraps MIKASA-Robo-VLA (ManiSkill/SAPIEN) memory-intensive manipulation tasks
into Gymnasium-compatible ``VectorEnv``s suitable for ``lerobot_eval``.

The benchmark contains 90 language-conditioned tasks across 10 memory types
(e.g. "ShellGameTouch-VLA-v0", "RememberColor9-VLA-v0"). The full manifest is
``mikasa_robo_vla_envs.csv`` in the MIKASA-Robo repository.

Observation/action contract (matches the released LeRobot v3 datasets,
https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot):
  - pixels/top:   base camera RGB (128, 128, 3)
  - pixels/wrist: hand camera RGB (128, 128, 3)
  - agent_pos:    eef xyz(3) + rpy(3) + gripper(1) proprio (7,)
  - action:       pd_ee_delta_pose, 7-D in [-1, 1]

Requires `mikasa_robo_suite` and `mani_skill` installed separately.
GPU simulation cannot be instantiated twice in one process, so use async
vector envs (subprocesses) when n_envs > 1.

Benchmark: https://mikasarobo.github.io/
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.envs.utils import _LazyAsyncVectorEnv

OBS_HEIGHT = 128
OBS_WIDTH = 128
ACTION_DIM = 7
PROPRIO_DIM = 7


class MikasaGymEnv(gym.Env):
    """Thin single-episode Gymnasium wrapper around one MIKASA-Robo-VLA env.

    The underlying ManiSkill env is created lazily on first ``reset()`` so
    that ``AsyncVectorEnv`` workers build their own simulation context instead
    of inheriting a stale one from the parent process.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task: str = "ShellGameTouch-VLA-v0",
        obs_type: str = "pixels_agent_pos",
        sim_backend: str = "gpu",
        render_backend: str = "gpu",
        control_mode: str = "pd_ee_delta_pose",
        reward_mode: str = "normalized_dense",
        max_episode_steps: int | None = None,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self._sim_backend = sim_backend
        self._render_backend = render_backend
        self._control_mode = control_mode
        self._reward_mode = reward_mode
        self._max_episode_steps_override = max_episode_steps
        self.task_description = ""  # populated once the env is created

        self._env = None

        if obs_type not in ("pixels", "pixels_agent_pos"):
            raise ValueError(
                f"Unsupported obs_type '{obs_type}'. "
                "MikasaGymEnv supports 'pixels' and 'pixels_agent_pos'."
            )

        # `pixels` must be a nested Dict so `preprocess_observation()` in
        # envs/utils.py maps each camera to `observation.images.<cam>`.
        obs_spaces: dict[str, spaces.Space] = {
            "pixels": spaces.Dict(
                {
                    "top": spaces.Box(0, 255, shape=(OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8),
                    "wrist": spaces.Box(0, 255, shape=(OBS_HEIGHT, OBS_WIDTH, 3), dtype=np.uint8),
                }
            )
        }
        if obs_type == "pixels_agent_pos":
            obs_spaces["agent_pos"] = spaces.Box(-np.inf, np.inf, shape=(PROPRIO_DIM,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

    @property
    def _device(self):
        return self._env.unwrapped.device

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        import mikasa_robo_suite.vla.memory_envs  # noqa: F401  (registers VLA env IDs)
        from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

        make_kwargs: dict[str, Any] = {
            "num_envs": 1,
            "obs_mode": "rgb",
            "control_mode": self._control_mode,
            "reward_mode": self._reward_mode,
            "render_mode": "all",
            "sim_backend": self._sim_backend,
            "render_backend": self._render_backend,
        }
        if self._max_episode_steps_override is not None:
            make_kwargs["max_episode_steps"] = self._max_episode_steps_override
        env = gym.make(self.task, **make_kwargs)
        env = apply_mikasa_vla_wrappers(env, include_overlays=False)
        self._env = env
        self._max_episode_steps = env.max_episode_steps
        self.task_description = str(getattr(env.unwrapped, "LANGUAGE_INSTRUCTION", self.task))

    def reset(self, *, seed=None, options=None):
        self._ensure_env()
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed)
        conv_info = self._convert_info(info)
        conv_info["is_success"] = False
        return self._convert_obs(obs), conv_info

    def step(self, action):
        import torch

        assert self._env is not None, "step() called before reset()"
        action_t = torch.as_tensor(
            np.asarray(action, dtype=np.float32).reshape(1, -1), device=self._device
        )
        obs, reward, terminated, truncated, info = self._env.step(action_t)

        reward_f = float(_to_numpy(reward).reshape(-1)[0])
        terminated_b = bool(_to_numpy(terminated).reshape(-1)[0])
        truncated_b = bool(_to_numpy(truncated).reshape(-1)[0])

        conv_info = self._convert_info(info)
        return self._convert_obs(obs), reward_f, terminated_b, truncated_b, conv_info

    def render(self) -> np.ndarray | None:
        """Return the combined ManiSkill render frame for video recording."""
        if self._env is None:
            return None
        frame = _to_numpy(self._env.render())
        if frame.ndim == 4:
            frame = frame[0]
        return np.ascontiguousarray(frame.astype(np.uint8, copy=False))

    def _convert_obs(self, obs: dict) -> dict:
        # After `apply_mikasa_vla_wrappers`, obs["rgb"] stacks the base and
        # hand cameras channel-wise: (1, H, W, 6) = [top(3) | wrist(3)].
        rgb = _to_numpy(obs["rgb"])[0].astype(np.uint8, copy=False)
        converted: dict[str, Any] = {
            "pixels": {
                "top": np.ascontiguousarray(rgb[..., :3]),
                "wrist": np.ascontiguousarray(rgb[..., 3:6]),
            }
        }
        if self.obs_type == "pixels_agent_pos":
            converted["agent_pos"] = _to_numpy(obs["proprio"])[0].astype(np.float32, copy=False)
        return converted

    def _convert_info(self, info: dict) -> dict:
        is_success = bool(_to_numpy(info.get("success", False)).reshape(-1)[0])
        return {
            "task": self.task,
            "task_description": self.task_description,
            "elapsed_steps": int(_to_numpy(info.get("elapsed_steps", 0)).reshape(-1)[0]),
            "is_success": is_success,
        }

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def create_mikasa_envs(
    task: str,
    n_envs: int = 1,
    gym_kwargs: dict[str, Any] | None = None,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Create vectorized MIKASA-Robo-VLA environments for evaluation.

    `task` may be a single env id (e.g. "ShellGameTouch-VLA-v0") or a
    comma-separated list. Each env id becomes its own suite in the returned
    mapping, with a single task_id=0 vec env of n_envs rollouts.

    Returns {suite_name: {task_id: VectorEnv}} matching lerobot's expected format.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of env factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_names = [t.strip() for t in task.split(",") if t.strip()]
    if not task_names:
        raise ValueError("`task` must contain at least one MIKASA-Robo-VLA env id.")

    is_async = env_cls is gym.vector.AsyncVectorEnv
    cached_obs_space: spaces.Space | None = None
    cached_act_space: spaces.Space | None = None
    cached_metadata: dict[str, Any] | None = None
    out: dict[str, dict[int, gym.vector.VectorEnv]] = {}
    for task_name in task_names:
        fns = [(lambda tn=task_name: MikasaGymEnv(task=tn, **gym_kwargs)) for _ in range(n_envs)]
        if is_async:
            lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
            if cached_obs_space is None:
                cached_obs_space = lazy.observation_space
                cached_act_space = lazy.action_space
                cached_metadata = lazy.metadata
            out[task_name] = {0: lazy}
        else:
            out[task_name] = {0: env_cls(fns)}
    return out
