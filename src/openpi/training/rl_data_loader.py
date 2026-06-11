"""Goal-conditioned / RL data loading supporting both LeRobot and RLDS/DROID datasets.

This module adds HER-style (hindsight) sampling to openpi's standard data pipeline.
For every sampled frame ``t`` (within an episode) it returns four observations:

- ``observation``        : the state at ``t`` (s_t)
- ``next_observation``   : the state at ``t + 1`` (s_{t+1})
- ``future_observation`` : a *random* future state ``s_f`` with ``t < f <= g``
- ``goal_observation``   : a *random* future "goal" state ``s_g`` with ``t < g < episode_end``

The goal index is drawn from a (truncated) geometric distribution controlled by
``gamma`` (larger gamma => goals further in the future). The future index is then
drawn uniformly in ``(t, g]`` (i.e. between the current frame and the goal), matching
the "future-then-goal" semantics.

All four observations flow through the *exact same* openpi transform pipeline
(repack -> data transforms -> normalize -> model transforms), so every modality
(state + all camera images) is processed identically to the policy input. Each frame
also carries an ``*_is_pad`` flag that is ``True`` when the requested index had to be
clamped to the episode boundary (e.g. when ``t`` is the last frame and no real future
exists) -- mask these out in your RL loss.

Notes / limitations:
- For LeRobot datasets: each sample fetches 4 frames, so image decode + transforms run
  ~4x per item. Use ``num_workers > 0`` to parallelize.
- For RLDS/DROID: goal sampling is done inside the TF pipeline (before flattening) so
  the streaming shuffle buffer still sees full trajectories. Prompt tokenization is
  redundant across the 4 frames; can be optimized later if needed.
"""

from collections.abc import Iterator
import dataclasses
import logging
from typing import TypedDict

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


class GoalConditionedBatch(TypedDict, total=False):
    observation: _model.Observation
    next_observation: _model.Observation
    future_observation: _model.Observation
    goal_observation: _model.Observation
    actions: at.Float[at.Array, "*b ah ad"]
    next_is_pad: at.Array
    future_is_pad: at.Array
    goal_is_pad: at.Array

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GoalSamplingConfig:
    """Controls how future/goal indices are sampled."""

    # Geometric discount for the goal horizon. The goal offset d (>=1) is drawn from a
    # geometric distribution with p = 1 - gamma, then clamped to the episode end.
    # gamma -> 0 picks near goals (mostly t+1); gamma -> 1 picks far goals (often episode end).
    gamma: float = 0.99
    # If True, the future index is sampled uniformly in (t, g]. If False, it uses the same
    # truncated-geometric scheme as the goal (restricted to (t, g]).
    future_uniform: bool = True
    # Base RNG seed. Combined with the worker id so each data-loader worker is independent.
    seed: int = 0


def _per_frame_episode_bounds(dataset: lerobot_dataset.LeRobotDataset) -> tuple[np.ndarray, np.ndarray]:
    """Build per-frame [episode_start, episode_end) arrays (episode_end is exclusive)."""
    ep_from = np.asarray(dataset.episode_data_index["from"]).astype(np.int64)
    ep_to = np.asarray(dataset.episode_data_index["to"]).astype(np.int64)
    num_frames = len(dataset)
    ep_start = np.zeros(num_frames, dtype=np.int64)
    ep_end = np.zeros(num_frames, dtype=np.int64)
    for start, end in zip(ep_from, ep_to, strict=True):
        ep_start[start:end] = start
        ep_end[start:end] = end
    return ep_start, ep_end


class RandomFutureDataset(_data_loader.Dataset):
    """Wraps a (transformed) LeRobot dataset to additionally sample next/future/goal frames.

    All temporal offsets (next, future, goal) are measured in units of ``action_chunk_size``
    frames so that each "step" corresponds to one complete action chunk execution.
    """

    def __init__(
        self,
        transformed_dataset: _data_loader.Dataset,
        ep_start: np.ndarray,
        ep_end: np.ndarray,
        *,
        sampling: GoalSamplingConfig,
        action_chunk_size: int = 1,
    ):
        self._dataset = transformed_dataset
        self._ep_start = ep_start
        self._ep_end = ep_end
        self._sampling = sampling
        self._action_chunk_size = action_chunk_size
        self._gen: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self._dataset)

    def _rng(self) -> np.random.Generator:
        # Lazily build a per-worker RNG (the dataset object is copied into each worker).
        if self._gen is None:
            info = torch.utils.data.get_worker_info()
            worker_id = 0 if info is None else info.id
            self._gen = np.random.default_rng([self._sampling.seed, worker_id])
        return self._gen

    def _sample_goal_offset_chunks(self, max_chunks: int, rng: np.random.Generator) -> int:
        """Sample a goal offset in [1, max_chunks] chunks from a truncated geometric distribution."""
        p = max(1e-6, 1.0 - self._sampling.gamma)
        d = int(rng.geometric(p))
        return min(max(d, 1), max_chunks)

    def _sample_indices(self, t: int) -> tuple[int, int, int, dict[str, np.bool_]]:
        ep_end = int(self._ep_end[t])
        rng = self._rng()
        C = self._action_chunk_size

        # Number of full action chunks available after t within this episode.
        num_chunks = (ep_end - 1 - t) // C

        # Next: the frame that follows after executing one full action chunk.
        next_idx = min(t + C, ep_end - 1)
        next_is_pad = (t + C) >= ep_end

        if num_chunks <= 0:
            pads = {
                "next_is_pad": np.bool_(True),
                "future_is_pad": np.bool_(True),
                "goal_is_pad": np.bool_(True),
            }
            return t, t, t, pads

        goal_offset_chunks = self._sample_goal_offset_chunks(num_chunks, rng)
        goal_idx = t + goal_offset_chunks * C

        if self._sampling.future_uniform:
            future_offset_chunks = int(rng.integers(1, goal_offset_chunks + 1))
        else:
            future_offset_chunks = self._sample_goal_offset_chunks(goal_offset_chunks, rng)
        future_idx = t + future_offset_chunks * C

        pads = {
            "next_is_pad": np.bool_(next_is_pad),
            "future_is_pad": np.bool_(False),
            "goal_is_pad": np.bool_(False),
        }
        return next_idx, future_idx, goal_idx, pads

    def __getitem__(self, index) -> dict:
        t = int(index)
        next_idx, future_idx, goal_idx, pads = self._sample_indices(t)

        anchor = self._dataset[t]
        actions = anchor.get("actions")

        def _obs_only(sample: dict) -> dict:
            # Drop the action chunk from the auxiliary frames -- only the observation is needed.
            return {k: v for k, v in sample.items() if k != "actions"}

        item = {
            "observation": _obs_only(anchor),
            "next_observation": _obs_only(self._dataset[next_idx]),
            "future_observation": _obs_only(self._dataset[future_idx]),
            "goal_observation": _obs_only(self._dataset[goal_idx]),
            **pads,
        }
        if actions is not None:
            item["actions"] = actions
        return item


class GoalConditionedDataLoader(_data_loader.DataLoader):
    """Yields batches of structured goal-conditioned observations."""

    def __init__(self, data_config: _config.DataConfig, data_loader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self) -> Iterator[dict]:
        for batch in self._data_loader:
            out = {
                "observation": _model.Observation.from_dict(batch["observation"]),
                "next_observation": _model.Observation.from_dict(batch["next_observation"]),
                "future_observation": _model.Observation.from_dict(batch["future_observation"]),
                "goal_observation": _model.Observation.from_dict(batch["goal_observation"]),
                "next_is_pad": batch["next_is_pad"],
                "future_is_pad": batch["future_is_pad"],
                "goal_is_pad": batch["goal_is_pad"],
            }
            if "actions" in batch:
                out["actions"] = batch["actions"]
            yield out


class IterableHERTransformedDataset(_data_loader.IterableDataset):
    """Applies a transform pipeline to all 4 observations in an HER-structured RLDS batch.

    The RLDS HER dataset yields pre-batched dicts with keys ``observation``,
    ``next_observation``, ``future_observation``, ``goal_observation``, ``actions``,
    ``prompt``, and ``*_is_pad`` flags.  This wrapper splits each batch into individual
    samples, applies the transform pipeline to each of the four observations
    independently (only the anchor gets actions), and re-stacks into a batch.
    """

    def __init__(self, dataset: _data_loader.IterableDataset, transform_fn, aux_transform_fn=None):
        self._dataset = dataset
        self._transform_fn = transform_fn
        self._aux_transform_fn = aux_transform_fn if aux_transform_fn is not None else transform_fn

    def __iter__(self):
        for batch in self._dataset:
            leaves = jax.tree.leaves(batch)
            if not leaves:
                continue
            batch_size = leaves[0].shape[0]
            results = []
            for i in range(batch_size):
                sample = jax.tree.map(lambda x: x[i], batch)  # noqa: B023
                results.append(self._transform_sample(sample))
            yield jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *results)

    def _transform_sample(self, sample: dict) -> dict:
        prompt = sample.get("prompt")

        def _build_obs_input(obs_dict, *, include_actions: bool) -> dict:
            inp = {"observation": obs_dict}
            if prompt is not None:
                inp["prompt"] = prompt
            if include_actions:
                inp["actions"] = sample["actions"]
            return inp

        anchor_out = self._transform_fn(_build_obs_input(sample["observation"], include_actions=True))
        actions = anchor_out.pop("actions", None)

        def _transform_aux(obs_dict) -> dict:
            return self._aux_transform_fn(_build_obs_input(obs_dict, include_actions=False))

        result = {
            "observation": anchor_out,
            "next_observation": _transform_aux(sample["next_observation"]),
            "future_observation": _transform_aux(sample["future_observation"]),
            "goal_observation": _transform_aux(sample["goal_observation"]),
            "next_is_pad": sample["next_is_pad"],
            "future_is_pad": sample["future_is_pad"],
            "goal_is_pad": sample["goal_is_pad"],
        }
        if actions is not None:
            result["actions"] = actions
        return result

    def __len__(self) -> int:
        return len(self._dataset)


def create_goal_conditioned_data_loader(
    config: _config.TrainConfig,
    *,
    sampling: GoalSamplingConfig | None = None,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> GoalConditionedDataLoader:
    """Create a goal-conditioned (HER-style) data loader.

    Each yielded batch is a dict with the following keys (B = batch size):
        - ``observation``        : ``Observation`` at frame t
        - ``next_observation``   : ``Observation`` at t+1
        - ``future_observation`` : ``Observation`` at a random f in (t, g]
        - ``goal_observation``   : ``Observation`` at a random goal g in (t, episode_end)
        - ``actions``            : action chunk at t, shape (B, action_horizon, action_dim)
        - ``next_is_pad`` / ``future_is_pad`` / ``goal_is_pad`` : (B,) bool masks that are
          True when the index was clamped to the episode boundary (mask these in your loss)

    Args:
        config: The training configuration (defines the dataset + model + transforms).
        sampling: Goal/future sampling parameters. Defaults to ``GoalSamplingConfig()``.
        sharding: JAX sharding for the produced arrays. Defaults to data-parallel.
        shuffle: Whether to shuffle frame order.
        num_batches: If set, limits the number of returned batches (loops the dataset).
        skip_norm_stats: Skip normalization (useful before norm stats are computed).
    """
    sampling = sampling or GoalSamplingConfig(seed=config.seed)

    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return _create_goal_conditioned_rlds_data_loader(
            config,
            data_config,
            sampling=sampling,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )

    if data_config.repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if data_config.repo_id == "fake":
        raise ValueError("Goal-conditioned sampling requires a real dataset (got repo_id='fake').")

    action_horizon = config.model.action_horizon

    # Build the raw LeRobot dataset directly so we keep a handle for episode boundaries.
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    raw_dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    base_dataset: _data_loader.Dataset = raw_dataset
    if data_config.prompt_from_task:
        base_dataset = _data_loader.TransformedDataset(
            raw_dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    transformed = _data_loader.transform_dataset(base_dataset, data_config, skip_norm_stats=skip_norm_stats)

    ep_start, ep_end = _per_frame_episode_bounds(raw_dataset)
    dataset = RandomFutureDataset(transformed, ep_start, ep_end, sampling=sampling, action_chunk_size=action_horizon)

    local_batch_size = config.batch_size // jax.process_count()
    logger.info(f"local_batch_size: {local_batch_size}")

    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    return GoalConditionedDataLoader(data_config, torch_loader)


def _create_goal_conditioned_rlds_data_loader(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    *,
    sampling: GoalSamplingConfig,
    sharding: jax.sharding.Sharding | None,
    shuffle: bool,
    num_batches: int | None,
    skip_norm_stats: bool,
) -> GoalConditionedDataLoader:
    """RLDS/DROID path for goal-conditioned data loading.

    HER sampling is performed inside the TF pipeline (before trajectory flattening)
    so that future frames remain accessible while still benefiting from the streaming
    shuffle buffer.
    """
    from openpi.training.droid_rlds_dataset import DroidRldsDataset

    rlds_dataset = DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=config.batch_size,
        datasets=data_config.datasets,
        shuffle=shuffle,
        action_chunk_size=config.model.action_horizon,
        action_space=data_config.action_space,
        her_gamma=sampling.gamma,
        num_parallel_reads=data_config.rlds_num_parallel_reads,
        num_parallel_calls=data_config.rlds_num_parallel_calls,
        shuffle_buffer_size=data_config.rlds_shuffle_buffer_size,
    )

    norm_stats: dict = {}
    if not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    transform_fn = _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )

    # Aux observations (next/future/goal) carry no actions, so strip "actions" from
    # the repack structure to avoid a KeyError inside RepackTransform.
    def _strip_actions(t):
        if isinstance(t, _transforms.RepackTransform) and isinstance(t.structure, dict):
            return dataclasses.replace(t, structure={k: v for k, v in t.structure.items() if k != "actions"})
        return t

    aux_transform_fn = _transforms.compose(
        [
            *[_strip_actions(t) for t in data_config.repack_transforms.inputs],
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )

    her_dataset = IterableHERTransformedDataset(rlds_dataset, transform_fn, aux_transform_fn)

    rlds_loader = _data_loader.RLDSDataLoader(
        her_dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return GoalConditionedDataLoader(data_config, rlds_loader)
