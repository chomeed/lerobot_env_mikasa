"""EnvConfig for the MIKASA-Robo-VLA benchmark, registered as `mikasa`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig, _make_vec_env_cls
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@EnvConfig.register_subclass("mikasa")
@dataclass
class MikasaEnv(EnvConfig):
    """MIKASA-Robo-VLA memory-intensive manipulation benchmark (ManiSkill/SAPIEN).

    90 language-conditioned tasks across 10 memory types, single-arm Panda
    with base ("top") + hand ("wrist") cameras and 7-D pd_ee_delta_pose actions.
    `task` is an env id (e.g. "ShellGameTouch-VLA-v0") or a comma-separated list.

    Dataset: https://huggingface.co/datasets/mikasa-robo/mikasa-robo-vla-lerobot
    Benchmark: https://mikasarobo.github.io/

    Requires `mikasa_robo_suite` and `mani_skill` installed separately.
    GPU simulation cannot be instantiated twice in one process, so keep
    async (subprocess) vector envs when n_envs > 1.
    """

    task: str = "ShellGameTouch-VLA-v0"
    fps: int = 10
    episode_length: int | None = None  # None -> the env's registered horizon
    obs_type: str = "pixels_agent_pos"
    sim_backend: str = "gpu"
    render_backend: str = "gpu"
    control_mode: str = "pd_ee_delta_pose"
    reward_mode: str = "normalized_dense"
    observation_height: int = 128
    observation_width: int = 128
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "agent_pos": OBS_STATE,
            "pixels/top": f"{OBS_IMAGES}.top",
            "pixels/wrist": f"{OBS_IMAGES}.wrist",
        }
    )

    def __post_init__(self):
        self.features["pixels/top"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
        )
        self.features["pixels/wrist"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(self.observation_height, self.observation_width, 3)
        )
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
        elif self.obs_type != "pixels":
            raise ValueError(
                f"Unsupported obs_type '{self.obs_type}'. "
                "MikasaEnv supports 'pixels' and 'pixels_agent_pos'."
            )

    @property
    def gym_kwargs(self) -> dict:
        kwargs: dict[str, Any] = {
            "obs_type": self.obs_type,
            "sim_backend": self.sim_backend,
            "render_backend": self.render_backend,
            "control_mode": self.control_mode,
            "reward_mode": self.reward_mode,
        }
        if self.episode_length is not None:
            kwargs["max_episode_steps"] = self.episode_length
        return kwargs

    def create_envs(self, n_envs: int, use_async_envs: bool = True):
        from lerobot_env_mikasa.mikasa import create_mikasa_envs

        if not self.task:
            raise ValueError("MikasaEnv requires `task` to be specified.")
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        return create_mikasa_envs(
            task=self.task,
            n_envs=n_envs,
            gym_kwargs=self.gym_kwargs,
            env_cls=env_cls,
        )
