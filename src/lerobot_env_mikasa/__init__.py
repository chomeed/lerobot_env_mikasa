"""LeRobot env plugin for the MIKASA-Robo-VLA benchmark.

Installing this package makes `--env.type=mikasa` available in lerobot CLI
scripts: lerobot's `register_third_party_plugins()` imports any installed
`lerobot_env_*` package, and importing this module registers `MikasaEnv`
with the `EnvConfig` choice registry.
"""

from lerobot_env_mikasa.configuration_mikasa import MikasaEnv
from lerobot_env_mikasa.mikasa import MikasaGymEnv, create_mikasa_envs

__all__ = ["MikasaEnv", "MikasaGymEnv", "create_mikasa_envs"]
