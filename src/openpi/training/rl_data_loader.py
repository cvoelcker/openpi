"""Goal-conditioned / RL data loading supporting both LeRobot and RLDS/DROID datasets.

This module adds HER-style (hindsight) sampling to openpi's standard data pipeline.
For every sampled frame ``t`` (within an episode) it returns four observations:

- ``observation``        : the state at ``t`` (s_t)
- ``next_observation``   : the state one FULL ACTION CHUNK later, ``t + action_chunk_size``
  (10 control steps for the LIBERO configs) — i.e. the frame reached after executing the
  ``actions`` chunk, NOT ``t + 1``. See ``_sample_indices``.
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
    # Explicit within-task negative: a frame sampled uniformly from any episode of the anchor's
    # task (possibly its own trajectory — deliberately unbiased). Serves as a hard negative
    # candidate future in the CRL contrast.
    negative_observation: _model.Observation
    actions: at.Float[at.Array, "*b ah ad"]
    next_actions: at.Float[at.Array, "*b ah ad"]
    next_is_pad: at.Array
    future_is_pad: at.Array
    goal_is_pad: at.Array
    # Per-sample episode identifier (the anchor frame's episode).
    episode_id: at.Array
    # Absolute frame indices of the drawn frames (for temporal-offset diagnostics such as the
    # val ranking probe), plus episode ids for frames that may leave the anchor's episode.
    frame_index: at.Array
    next_frame_index: at.Array
    future_frame_index: at.Array
    future_episode_id: at.Array
    goal_frame_index: at.Array
    negative_frame_index: at.Array
    negative_episode_id: at.Array


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class GoalSamplingConfig:
    """Controls how future/goal indices are sampled."""

    # Geometric discount for the goal horizon. The goal offset d (>=1) is drawn from a
    # geometric distribution with p = 1 - gamma, then clamped to the episode end.
    # gamma -> 0 picks near goals (mostly t+1); gamma -> 1 picks far goals (often episode end).
    gamma: float = 0.99
    # Geometric discount for the CRL *positive* future. The future offset is drawn directly
    # from a geometric truncated (renormalized) to the remaining episode -- decoupled from the
    # goal, so the positive's marginal is a true geometric (LIBERO episodes are short, so 0.95
    # spreads the positive across the episode without piling mass at the boundary).
    future_gamma: float = 0.95
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


def _per_frame_task_index(dataset: lerobot_dataset.LeRobotDataset) -> np.ndarray:
    """Read the per-frame ``task_index`` column (metadata only, no image decode)."""
    hf = dataset.hf_dataset
    try:
        # Arrow column access never touches the (lazily-decoded) image columns.
        return np.asarray(hf.data.column("task_index").to_numpy(zero_copy_only=False)).astype(np.int64)
    except Exception:  # noqa: BLE001 -- fall back to the datasets column accessor
        return np.asarray(hf["task_index"]).astype(np.int64)


def _build_task_to_frames(frame_task: np.ndarray, frames: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """Map each task_index to the frame indices belonging to it. Built once at init. When
    ``frames`` is given, restrict to that subset (used to keep train/val negatives within-split)."""
    frames = np.arange(len(frame_task)) if frames is None else np.asarray(frames)
    tasks = frame_task[frames]
    return {int(task): frames[tasks == task] for task in np.unique(tasks)}


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
        include_next_observation: bool = True,
        include_future_observation: bool = True,
        include_goal_observation: bool = True,
        include_negative_observation: bool = False,
        random_future_control: bool = False,
        frame_task: np.ndarray | None = None,
        task_to_frames: dict[int, np.ndarray] | None = None,
    ):
        self._dataset = transformed_dataset
        self._ep_start = ep_start
        self._ep_end = ep_end
        self._sampling = sampling
        self._action_chunk_size = action_chunk_size
        self._include_next = include_next_observation
        self._include_future = include_future_observation
        self._include_goal = include_goal_observation
        self._include_negative = include_negative_observation
        self._random_future_control = random_future_control
        if include_negative_observation and (frame_task is None or task_to_frames is None):
            raise ValueError("include_negative_observation=True requires frame_task and task_to_frames.")
        self._frame_task = frame_task
        self._task_to_frames = task_to_frames
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
        """Sample a goal offset in [1, max_chunks] chunks from a (clamped) geometric distribution."""
        p = max(1e-6, 1.0 - self._sampling.gamma)
        d = int(rng.geometric(p))
        return min(max(d, 1), max_chunks)

    @staticmethod
    def _sample_truncated_geometric(max_chunks: int, gamma: float, rng: np.random.Generator) -> int:
        """Sample an offset in [1, max_chunks] from a geometric *truncated* (renormalized) to that
        range: P(d) ∝ (1 - gamma) * gamma**(d - 1). Unlike clamping, this does not pile the tail
        mass on the boundary offset -- it renormalizes over [1, max_chunks] via the inverse CDF.
        gamma -> 0 favors near offsets; gamma -> 1 approaches uniform over the range."""
        if max_chunks <= 1:
            return 1
        g = min(max(gamma, 1e-6), 1.0 - 1e-6)
        # F(k) = (1 - g**k) / (1 - g**max_chunks); solve for the smallest k with F(k) >= u.
        cdf_max = 1.0 - g**max_chunks
        u = rng.random()
        k = int(np.ceil(np.log1p(-u * cdf_max) / np.log(g)))
        return min(max(k, 1), max_chunks)

    def _control_future_index(self, t: int) -> int:
        """Randomization-test positive: a random frame from a DIFFERENT episode, deterministic in
        the anchor index (stable across epochs/workers) so the pairing is memorizable but carries
        no temporal signal. Uniform over all out-of-episode frames via index remapping (no
        rejection loop). Falls back to t itself for a degenerate single-episode dataset."""
        ep_start, ep_end = int(self._ep_start[t]), int(self._ep_end[t])
        num_valid = len(self._dataset) - (ep_end - ep_start)
        if num_valid <= 0:
            return t
        # Salted, anchor-keyed RNG: independent of the per-worker sampling stream.
        rng = np.random.default_rng([self._sampling.seed, 0x5EED, t])
        u = int(rng.integers(0, num_valid))
        return u if u < ep_start else u + (ep_end - ep_start)

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

        # CRL positive: geometric future over the whole remaining episode, decoupled from the
        # goal so its marginal is a true (truncated) geometric rather than a goal-conditioned mix.
        future_offset_chunks = self._sample_truncated_geometric(num_chunks, self._sampling.future_gamma, rng)
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
        if self._random_future_control:
            # Override only the CRL positive; next/goal/negative keep their real semantics. A
            # cross-episode partner always exists, so the future is never a boundary-clamped pad.
            future_idx = self._control_future_index(t)
            pads["future_is_pad"] = np.bool_(future_idx == t)

        anchor = self._dataset[t]
        actions = anchor.get("actions")

        def _obs_only(sample: dict) -> dict:
            return {k: v for k, v in sample.items() if k != "actions"}

        # The episode-start frame index uniquely identifies the anchor's episode; equal
        # values mean same episode. Absolute frame indices for every drawn frame are also
        # emitted so downstream diagnostics (e.g. the val ranking probe) can reconstruct
        # temporal offsets and same-episode relations without re-touching the dataset.
        item = {
            "observation": _obs_only(anchor),
            "episode_id": np.int64(self._ep_start[t]),
            "frame_index": np.int64(t),
        }
        if self._include_next:
            item["next_observation"] = _obs_only(self._dataset[next_idx])
            item["next_is_pad"] = pads["next_is_pad"]
            item["next_frame_index"] = np.int64(next_idx)
        if self._include_future:
            item["future_observation"] = _obs_only(self._dataset[future_idx])
            item["future_is_pad"] = pads["future_is_pad"]
            item["future_frame_index"] = np.int64(future_idx)
            # Same as episode_id except in random_future_control mode (cross-episode positive).
            item["future_episode_id"] = np.int64(self._ep_start[future_idx])
        if self._include_goal:
            item["goal_observation"] = _obs_only(self._dataset[goal_idx])
            item["goal_is_pad"] = pads["goal_is_pad"]
            item["goal_frame_index"] = np.int64(goal_idx)
        if self._include_negative:
            # Explicit within-task negative: uniform over all frames sharing the anchor's task,
            # deliberately unbiased (may land on the anchor's own trajectory).
            same_task_frames = self._task_to_frames[int(self._frame_task[t])]
            neg_idx = int(same_task_frames[self._rng().integers(0, len(same_task_frames))])
            item["negative_observation"] = _obs_only(self._dataset[neg_idx])
            item["negative_frame_index"] = np.int64(neg_idx)
            item["negative_episode_id"] = np.int64(self._ep_start[neg_idx])
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
            out = {"observation": _model.Observation.from_dict(batch["observation"])}
            for key in ("next_observation", "future_observation", "goal_observation", "negative_observation"):
                if key in batch:
                    out[key] = _model.Observation.from_dict(batch[key])
            for key in (
                "next_is_pad",
                "future_is_pad",
                "goal_is_pad",
                "episode_id",
                "frame_index",
                "next_frame_index",
                "future_frame_index",
                "future_episode_id",
                "goal_frame_index",
                "negative_frame_index",
                "negative_episode_id",
            ):
                if key in batch:
                    out[key] = batch[key]
            if "actions" in batch:
                out["actions"] = batch["actions"]
            if "next_actions" in batch:
                out["next_actions"] = batch["next_actions"]
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

        def _build_obs_input(obs_dict, *, actions=None) -> dict:
            inp = {"observation": obs_dict}
            if prompt is not None:
                inp["prompt"] = prompt
            if actions is not None:
                # RLDS actions are read-only; DeltaActions/AbsoluteActions mutate in place
                # (upstream dropped their internal .copy()), so hand them a writable copy.
                inp["actions"] = np.array(actions)
            return inp

        anchor_out = self._transform_fn(_build_obs_input(sample["observation"], actions=sample["actions"]))
        actions = anchor_out.pop("actions", None)

        def _transform_aux(obs_dict) -> dict:
            return self._aux_transform_fn(_build_obs_input(obs_dict))

        # next_observation is transformed through the anchor pipeline together with its
        # action chunk, so next_actions is normalized identically to actions (SARSA a'
        # for successor-feature training). future/goal carry no actions, as before.
        next_actions = None
        next_observation = None
        if "next_observation" in sample:
            if "next_actions" in sample:
                next_out = self._transform_fn(
                    _build_obs_input(sample["next_observation"], actions=sample["next_actions"])
                )
                next_actions = next_out.pop("actions", None)
                next_observation = next_out
            else:
                next_observation = _transform_aux(sample["next_observation"])

        result = {"observation": anchor_out}
        if next_observation is not None:
            result["next_observation"] = next_observation
        for key in ("future_observation", "goal_observation"):
            if key in sample:
                result[key] = _transform_aux(sample[key])
        for key in ("next_is_pad", "future_is_pad", "goal_is_pad", "episode_id"):
            if key in sample:
                result[key] = sample[key]
        if actions is not None:
            result["actions"] = actions
        if next_actions is not None:
            result["next_actions"] = next_actions
        return result

    def __len__(self) -> int:
        return len(self._dataset)


def _process_sharded_sampler(indices: np.ndarray, *, shuffle: bool, seed: int) -> torch.utils.data.Sampler | list[int]:
    """Sampler over `indices` restricted to this process's (rank-strided) shard.

    The torch loaders here feed jax.make_array_from_process_local_data, which assembles the
    global batch from each process's local batch — so every process must draw a disjoint
    slice of the data. Without this, the identically-seeded loaders yield the same local
    batch on every process and the "global" batch is jax.process_count() copies of it.
    A persistent seeded generator draws a fresh permutation every epoch while staying
    reproducible across runs.
    """
    local = indices[jax.process_index() :: jax.process_count()].tolist()
    if not shuffle:
        return local
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.SubsetRandomSampler(local, generator=generator)


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
        - ``next_observation``   : ``Observation`` at t + action_chunk_size (one full chunk)
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
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    base_dataset: _data_loader.Dataset = raw_dataset
    if data_config.prompt_from_task:
        base_dataset = _data_loader.TransformedDataset(
            raw_dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    transformed = _data_loader.transform_dataset(base_dataset, data_config, skip_norm_stats=skip_norm_stats)

    ep_start, ep_end = _per_frame_episode_bounds(raw_dataset)
    frame_task, task_to_frames = (None, None)
    if data_config.include_negative_observation:
        frame_task = _per_frame_task_index(raw_dataset)
        task_to_frames = _build_task_to_frames(frame_task)
    dataset = RandomFutureDataset(
        transformed,
        ep_start,
        ep_end,
        sampling=sampling,
        action_chunk_size=action_horizon,
        include_next_observation=data_config.include_next_observation,
        include_future_observation=data_config.include_future_observation,
        include_goal_observation=data_config.include_goal_observation,
        include_negative_observation=data_config.include_negative_observation,
        random_future_control=data_config.random_future_control,
        frame_task=frame_task,
        task_to_frames=task_to_frames,
    )

    local_batch_size = config.batch_size // jax.process_count()
    logger.info(f"local_batch_size: {local_batch_size}")

    # Multi-process: each process samples a disjoint shard (single-process keeps the
    # plain shuffle path so existing single-GPU runs draw the exact same batches).
    sampler = None
    if jax.process_count() > 1:
        sampler = _process_sharded_sampler(np.arange(len(dataset)), shuffle=shuffle, seed=config.seed)

    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    return GoalConditionedDataLoader(data_config, torch_loader)


def create_train_val_goal_conditioned_data_loaders(
    config: _config.TrainConfig,
    *,
    sampling: GoalSamplingConfig | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> tuple[GoalConditionedDataLoader, GoalConditionedDataLoader | None]:
    """Create train and (optionally) val goal-conditioned data loaders with episode-level splitting.

    Returns (train_loader, val_loader). val_loader is None when config.val_fraction == 0.
    """
    if config.val_fraction <= 0:
        train_loader = create_goal_conditioned_data_loader(config, sampling=sampling, sharding=sharding, shuffle=True)
        return train_loader, None

    sampling = sampling or GoalSamplingConfig(seed=config.seed)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        val_pct = int(config.val_fraction * 100)
        train_loader = _create_goal_conditioned_rlds_data_loader(
            config,
            data_config,
            sampling=sampling,
            sharding=sharding,
            shuffle=True,
            num_batches=None,
            skip_norm_stats=False,
            tfds_split=f"train[:{100 - val_pct}%]",
        )
        val_loader = _create_goal_conditioned_rlds_data_loader(
            config,
            data_config,
            sampling=sampling,
            sharding=sharding,
            shuffle=False,
            num_batches=None,
            skip_norm_stats=False,
            tfds_split=f"train[{100 - val_pct}%:]",
        )
        return train_loader, val_loader

    # LeRobot path: split by episode indices.
    if data_config.repo_id is None or data_config.repo_id == "fake":
        raise ValueError("Val split requires a real dataset.")

    action_horizon = config.model.action_horizon
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    raw_dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    base_dataset: _data_loader.Dataset = raw_dataset
    if data_config.prompt_from_task:
        base_dataset = _data_loader.TransformedDataset(
            raw_dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    transformed = _data_loader.transform_dataset(base_dataset, data_config)

    ep_start, ep_end = _per_frame_episode_bounds(raw_dataset)
    train_indices, val_indices = _data_loader._split_episode_indices(raw_dataset, config.val_fraction, config.seed)

    frame_task = _per_frame_task_index(raw_dataset) if data_config.include_negative_observation else None

    def _make_loader(indices):
        # Restrict negatives to the anchor's own split so the val negative marginal stays clean.
        split_task_to_frames = (
            _build_task_to_frames(frame_task, np.asarray(indices)) if data_config.include_negative_observation else None
        )
        dataset = RandomFutureDataset(
            transformed,
            ep_start,
            ep_end,
            sampling=sampling,
            action_chunk_size=action_horizon,
            include_next_observation=data_config.include_next_observation,
            include_future_observation=data_config.include_future_observation,
            include_goal_observation=data_config.include_goal_observation,
            include_negative_observation=data_config.include_negative_observation,
            random_future_control=data_config.random_future_control,
            frame_task=frame_task,
            task_to_frames=split_task_to_frames,
        )
        local_batch_size = config.batch_size // jax.process_count()
        # Random order for train AND val (see the num_workers comment below for why val
        # samples randomly too), each process drawing a disjoint shard of the split.
        sampler = _process_sharded_sampler(np.asarray(indices), shuffle=True, seed=config.seed)
        # Use the same worker count for train and val. With per-worker RNG seeding
        # (`_rng` = default_rng([seed, worker_id])), a val loader pinned to num_workers=0 would
        # draw all its futures/goals/negatives from a single persistent stream — lower sampling
        # diversity than train's N streams, which biases the train/val rep-loss comparison
        # independent of true generalization. Matching worker counts removes that confound.
        torch_loader = _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=sharding,
            sampler=sampler,
            num_workers=config.num_workers,
            seed=config.seed,
        )
        return GoalConditionedDataLoader(data_config, torch_loader)

    return _make_loader(train_indices), _make_loader(val_indices)


def _create_goal_conditioned_rlds_data_loader(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    *,
    sampling: GoalSamplingConfig,
    sharding: jax.sharding.Sharding | None,
    shuffle: bool,
    num_batches: int | None,
    skip_norm_stats: bool,
    tfds_split: str = "train",
) -> GoalConditionedDataLoader:
    """RLDS/DROID path for goal-conditioned data loading.

    HER sampling is performed inside the TF pipeline (before trajectory flattening)
    so that future frames remain accessible while still benefiting from the streaming
    shuffle buffer.
    """
    from openpi.training.droid_rlds_dataset import DroidRldsDataset

    if data_config.random_future_control:
        raise NotImplementedError("random_future_control is only implemented for the LeRobot path.")

    local_batch_size = config.batch_size // jax.process_count()
    logger.info(f"RLDS local_batch_size: {local_batch_size}")

    rlds_dataset = DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=local_batch_size,
        datasets=data_config.datasets,
        shuffle=shuffle,
        action_chunk_size=config.model.action_horizon,
        action_space=data_config.action_space,
        her_gamma=sampling.gamma,
        num_parallel_reads=data_config.rlds_num_parallel_reads,
        num_parallel_calls=data_config.rlds_num_parallel_calls,
        tfds_split=tfds_split,
        shuffle_buffer_size=data_config.rlds_shuffle_buffer_size,
        include_next_observation=data_config.include_next_observation,
        include_future_observation=data_config.include_future_observation,
        include_goal_observation=data_config.include_goal_observation,
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
