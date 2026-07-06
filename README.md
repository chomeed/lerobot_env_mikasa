# lerobot_env_mikasa

LeRobot env plugin for the [MIKASA-Robo-VLA](https://mikasarobo.github.io/) memory-intensive
manipulation benchmark (ManiSkill/SAPIEN). Installing it makes `--env.type=mikasa` available
in `lerobot-train` / `lerobot-eval` via lerobot's third-party plugin discovery — no fork of
lerobot needed.

## Install

Into an env that already has lerobot, mikasa_robo_suite, and mani_skill:

```bash
pip install -e . --no-deps
```

Note: mikasa_robo_suite's wrappers require `gymnasium==0.29.1` (gymnasium 1.0
removed the `Wrapper.__getattr__` attribute forwarding they rely on), which
conflicts with lerobot's declared deps. Installing with deps lets pip upgrade
gymnasium and breaks MIKASA env creation — hence `--no-deps`.

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
