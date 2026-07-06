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

`--no-deps` is required: the mikasa stack needs `gymnasium==0.29.1` / `numpy<2`
(pinned in this package's dependencies), which conflicts with lerobot's declared
deps — a with-deps install fails with `ResolutionImpossible` on purpose rather
than letting pip silently upgrade gymnasium and break MIKASA env creation.
To protect the env from *future* pip installs moving these versions, see
[constraints.txt](constraints.txt).

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
