# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

openpi is Physical Intelligence's open-source robotics repo containing three Vision-Language-Action (VLA) models:
- **π₀ (pi0)**: Flow-based VLA model
- **π₀-FAST (pi0_fast)**: Autoregressive VLA using the FAST action tokenizer
- **π₀.₅ (pi05)**: Upgraded π₀ with better open-world generalization via knowledge insulation

The repo has dual JAX (primary) and PyTorch (secondary) implementations. Models are pre-trained on 10k+ hours of robot data and support fine-tuning.

## Development Commands

```bash
# Install dependencies (requires uv)
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest src/openpi/models/pi0_test.py

# Run a single test
uv run pytest src/openpi/models/pi0_test.py::test_name

# Linting and formatting
ruff check .
ruff format .

# Pre-commit (run before PRs)
pre-commit install  # one-time setup
pre-commit run --all-files

# Compute normalization stats before training
uv run scripts/compute_norm_stats.py --config-name pi05_libero

# Train (JAX)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite

# Train (PyTorch, single GPU)
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Serve a policy
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

## Architecture

### Package Structure

- `src/openpi/` — main library
  - `models/` — JAX model implementations (Gemma, SigLIP ViT, pi0, pi0_fast, LoRA, tokenizer)
  - `models_pytorch/` — PyTorch equivalents (pi0, Gemma, preprocessing)
  - `policies/` — robot-specific input/output transforms and policy wrappers (`aloha_policy.py`, `droid_policy.py`, `libero_policy.py`)
  - `training/` — training loop, data loading, checkpointing, optimizer, config definitions
  - `serving/` — WebSocket policy server (`websocket_policy_server.py`)
  - `shared/` — utilities: array typing, normalization, download, image tools
  - `transforms.py` — core `DataTransformFn` protocol and `Group` composition
- `packages/openpi-client/` — lightweight client package (minimal deps, Python 3.7+) for connecting robots to the policy server via WebSocket
- `scripts/` — entry points: `train.py`, `train_pytorch.py`, `serve_policy.py`, `compute_norm_stats.py`
- `examples/` — robot-specific examples (aloha_sim, aloha_real, droid, libero, ur5)

### Data Flow

1. **Input transforms** (`DataTransformFn`): robot-specific observation → normalized model input dict with keys `image`, `image_mask`, `state`, `actions`, `prompt`
2. **Model** (`BaseModel`): takes `Observation` + `state` → `Actions` (JAX uses Flax NNX, PyTorch uses `nn.Module`)
3. **Output transforms**: model actions → robot-specific action format
4. **Policy** (`Policy` class in `policies/policy.py`): wraps model + transforms, exposes `.infer(obs)` → `{"actions": ...}`

### Config System

`src/openpi/training/config.py` is the central config registry. `TrainConfig` composes:
- `ModelConfig` — architecture (pi0, pi0_fast, pi05) and hyperparameters
- `DataConfig` — LeRobot repo ID, transform groups, norm stats, action sequence keys
- `AssetsConfig` — where to load pre-computed normalization statistics from

Named configs (e.g., `"pi05_libero"`, `"pi0_aloha_sim"`) are registered in `_CONFIGS` at the bottom of `config.py`. Use `_config.get_config("name")` to retrieve them.

### Client/Server Split

`openpi-client` package (in `packages/`) is intentionally minimal (Python 3.7+, no JAX/PyTorch). Robots run `WebsocketClientPolicy` which sends observations to the server and receives actions. The server runs `WebsocketPolicyServer` in `src/openpi/serving/`. This separation keeps robot environments clean.

### Adding a New Robot/Dataset

Follow `src/openpi/policies/libero_policy.py` as a template:
1. Create `XxxInputs(transforms.DataTransformFn)` and `XxxOutputs(transforms.DataTransformFn)` for your robot
2. Add `LeRobotXxxDataConfig` and `TrainConfig` entries in `config.py`
3. Run `compute_norm_stats.py` before training

### PyTorch Setup Note

PyTorch support requires patching the installed `transformers` library:
```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```
This is required to support AdaRMS, activation precision control, and KV cache modifications. To undo: `uv cache clean transformers`.

### JAX Memory

Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before training to allow JAX to use 90% of GPU memory (default is 75%). Use `--fsdp-devices <n>` to enable fully-sharded data parallelism across `n` GPUs.

### Checkpoints

Checkpoints auto-download from `gs://openpi-assets` and cache in `~/.cache/openpi`. Override with `OPENPI_DATA_HOME` env var. PyTorch checkpoints are detected by presence of `model.safetensors`; JAX checkpoints have a `params/` directory.
