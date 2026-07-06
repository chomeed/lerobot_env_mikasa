# lerobot_env_mikasa

LeRobot env plugin for the [MIKASA-Robo-VLA](https://mikasarobo.github.io/) memory-intensive
manipulation benchmark (ManiSkill/SAPIEN). Installing it makes `--env.type=mikasa` available
in `lerobot-train` / `lerobot-eval` via lerobot's third-party plugin discovery — no fork of
lerobot needed.

> [!WARNING]
> **Version mismatch: `gymnasium` must stay at `0.29.1`, even though lerobot declares
> `gymnasium>=1.1.1`.** mikasa_robo_suite's wrappers rely on the implicit
> `Wrapper.__getattr__` attribute forwarding that gymnasium 1.0 removed — on gymnasium ≥1.0,
> MIKASA env creation crashes with
> `AttributeError: 'StateOnlyTensorToDictWrapper' object has no attribute 'device'`.
> Because of this conflict, always install this package (and anything else in the env) with
> `pip install --no-deps` so pip never re-resolves lerobot's deps and upgrades gymnasium.
> If gymnasium gets upgraded by accident, fix with `pip install gymnasium==0.29.1`.
> numpy is not affected: both 1.26.x and 2.x work with this stack.

## Install

Into an env that already has lerobot, mikasa_robo_suite, and mani_skill:

```bash
pip install -e . --no-deps
```

`--no-deps` matters — see the version-mismatch warning above.

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
