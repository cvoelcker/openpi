# openpi — Data Pipeline & How to Update the Dataloader

This document explains the basic structure of the repo and, in detail, how the
data loading pipeline works so you can adapt it to your own dataset.

## Repo layout (the parts that matter for data)

- `src/openpi/training/config.py` — All training configs (`TrainConfig`) and
  data configs (`DataConfig` + `DataConfigFactory` subclasses). This is the main
  file you edit to point training at a new dataset.
- `src/openpi/training/data_loader.py` — The actual dataloader implementation:
  dataset creation, transform application, and the JAX/PyTorch loaders.
- `src/openpi/training/droid_rlds_dataset.py` — Special-case streaming dataset
  for DROID (RLDS/TFDS format).
- `src/openpi/transforms.py` — Generic transform building blocks (repack,
  normalize, resize, tokenize, delta/absolute actions, etc.).
- `src/openpi/policies/*.py` — Dataset/robot-specific input & output transforms
  (e.g. `libero_policy.py`, `aloha_policy.py`, `droid_policy.py`).
- `scripts/train.py` / `scripts/train_pytorch.py` — Training entry points that
  call `create_data_loader`.
- `scripts/compute_norm_stats.py` — Precomputes normalization statistics that the
  dataloader requires.
- `examples/*/convert_*_to_lerobot.py` — Scripts converting raw data into the
  LeRobot dataset format the loader expects.

## How the dataloader works

The entry point is `create_data_loader(config, ...)` in
`src/openpi/training/data_loader.py`. The flow:

1. **Build the `DataConfig`.** `config.data.create(...)` runs the
   `DataConfigFactory` for your config, producing a `DataConfig` with the
   `repo_id`, normalization stats, and the four transform groups.
2. **Pick a backend** based on the config:
   - If `rlds_data_dir` is set → `create_rlds_data_loader` (DROID streaming, JAX only).
   - Otherwise → `create_torch_data_loader` (LeRobot datasets, JAX or PyTorch).
3. **Create the dataset.**
   - `create_torch_dataset` loads a `LeRobotDataset` by `repo_id`, using
     `delta_timestamps` to fetch an action chunk of length `action_horizon`.
     A `repo_id` of `"fake"` returns a `FakeDataset` of random samples.
   - `create_rlds_dataset` builds a `DroidRldsDataset`.
4. **Apply transforms** (`transform_dataset` / `transform_iterable_dataset`).
   Transforms are composed and applied in this fixed order:
   1. `repack_transforms.inputs` — rename raw dataset keys to a common format.
   2. `data_transforms.inputs` — robot/dataset-specific input conversion.
   3. `Normalize(norm_stats)` — z-score or quantile normalization.
   4. `model_transforms.inputs` — model-specific steps (resize images, tokenize
      prompt, pad states/actions).
5. **Wrap in a loader.** `TorchDataLoader` (or `RLDSDataLoader`) handles batching,
   shuffling, workers, and sharding. Batches are converted to sharded JAX arrays
   (or torch tensors for PyTorch). `DataLoaderImpl.__iter__` finally yields
   `(Observation, actions)` tuples.

### The four transform groups (key mental model)

Defined on `DataConfig` in `config.py`:

| Group | When applied | Used at inference? | Purpose |
|-------|-------------|--------------------|---------|
| `repack_transforms` | first | **No** (training only) | Rename dataset keys to match inference keys |
| `data_transforms`   | second | Yes | Robot-specific input/output conversion (e.g. `LiberoInputs`/`LiberoOutputs`) |
| `Normalize`         | third | Yes | Normalize state/actions using precomputed stats |
| `model_transforms`  | fourth | Yes | Resize images, tokenize prompt, pad to action dim |

`inputs` run on the way *into* the model; `outputs` (on `data_transforms` /
`model_transforms`) run in reverse on the way *out* during inference.

## How to update the dataloader for your own dataset

In most cases you do **not** edit `data_loader.py` itself — you add a new
`DataConfigFactory` and a `TrainConfig`. The cleanest template to copy is
`LeRobotLiberoDataConfig` (it has step-by-step comments).

### Step 1 — Convert your data to LeRobot format
Use one of the `examples/*/convert_*_to_lerobot.py` scripts as a template and
push/register it under a `repo_id`.

### Step 2 — Define dataset-specific input/output transforms
Copy `src/openpi/policies/libero_policy.py` to a new file (e.g.
`my_policy.py`). Edit `LiberoInputs` so it maps your dataset's keys into the
model format:

- `state` — proprioceptive state vector
- `image.{base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}` — camera views
  (pad missing cameras with zeros and set the matching `image_mask`)
- `actions` — present only during training
- `prompt` — language instruction

Edit `LiberoOutputs` to slice the model's padded action output back down to your
real action dimension (replace the `7` with your `action_dim`).

### Step 3 — Add a `DataConfigFactory`
In `config.py`, copy `LeRobotLiberoDataConfig` and adjust:

- **`repack_transforms`** — a `RepackTransform` mapping raw dataset keys
  (left side) to the keys your input transform expects (right side).
- **`data_transforms`** — point `inputs`/`outputs` at your new
  `MyInputs`/`MyOutputs`.
- **delta vs absolute actions** — if your dataset stores absolute actions, add
  `DeltaActions`/`AbsoluteActions` with a `make_bool_mask(...)` selecting which
  dims are converted (gripper dims usually stay absolute). pi0 expects deltas.
- **`action_sequence_keys`** — set if your action key isn't `"actions"`
  (e.g. Aloha uses `"action"`).
- **`model_transforms`** — usually leave as `ModelTransformFactory()`.

### Step 4 — Add a `TrainConfig`
Add an entry to the `_CONFIGS` list with your `name`, `model`, `data` (your new
factory + `repo_id`), `weight_loader`, and hyperparameters. Set
`prompt_from_task=True` in the `base_config` to pull the prompt from the LeRobot
`task` field.

### Step 5 — Compute normalization stats
The loader raises if `norm_stats` is missing. Run:

```bash
uv run scripts/compute_norm_stats.py --config-name=<your_config_name>
```

### Step 6 — Train
```bash
uv run scripts/train.py <your_config_name> --exp-name=my_experiment
```

## Things to watch out for

- **Key matching:** the right-hand side of your `RepackTransform` must match the
  keys your input transform reads, and those should mirror what your inference
  environment sends to the policy server.
- **Action dim / horizon:** set `action_dim` and `action_horizon` on the model
  config to match your dataset; outputs transform must un-pad to the real dim.
- **RLDS path is JAX-only** and requires `num_workers=0` (it handles its own
  multiprocessing). It is currently DROID-specific.
- **Norm stats are required** unless `repo_id="fake"` or `skip_norm_stats=True`.
- **Custom raw dataset (non-LeRobot, non-RLDS):** you'd implement the `Dataset`
  protocol (`__getitem__`/`__len__`) and add a branch in `create_torch_dataset`
  — but converting to LeRobot format first is strongly preferred.
