# lerobot_env_mikasa

LeRobot env plugin for the [MIKASA-Robo-VLA](https://mikasarobo.github.io/) memory-intensive
manipulation benchmark (ManiSkill/SAPIEN). Installing it makes `--env.type=mikasa` available
in `lerobot-train` / `lerobot-eval` via lerobot's third-party plugin discovery — no fork of
lerobot needed.

## Install

```bash
pip install -e .
# plus, separately (e.g. from envs/MIKASA-Robo):
#   mikasa_robo_suite, mani_skill
```

## Usage

```bash
lerobot-eval \
  --env.type=mikasa \
  --env.task=ShellGameTouch-VLA-v0 \
  ...
```

`--env.task` accepts a single env id or a comma-separated list. GPU simulation cannot be
instantiated twice in one process, so keep async (subprocess) vector envs when `n_envs > 1`.

## Layout

- `configuration_mikasa.py` — `MikasaEnv` (`EnvConfig` subclass, registered as `"mikasa"`)
- `mikasa.py` — `MikasaGymEnv` Gymnasium wrapper + `create_mikasa_envs` vec-env factory
