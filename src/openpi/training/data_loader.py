from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class MixedLeRobotDataset(Dataset):
    """Several LeRobot datasets concatenated behind one global frame index.

    Prompts are resolved per source, because ``task_index`` is dataset-local and only that
    source's metadata can decode it. Episode boundaries, task indices and the per-frame source
    id are re-based onto the global index so downstream episode splitting and HER sampling can
    treat the mixture as a single contiguous dataset.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        raw_datasets: Sequence[lerobot_dataset.LeRobotDataset],
        weights: Sequence[float] = (),
    ):
        self._datasets = list(datasets)
        self._raw_datasets = list(raw_datasets)
        lengths = [len(d) for d in self._datasets]
        self._offsets = np.cumsum([0, *lengths]).astype(np.int64)
        self._length = int(self._offsets[-1])

        ep_from, ep_to, episode_source = [], [], []
        for i, raw in enumerate(self._raw_datasets):
            offset = self._offsets[i]
            starts = np.asarray(raw.episode_data_index["from"]).astype(np.int64) + offset
            ends = np.asarray(raw.episode_data_index["to"]).astype(np.int64) + offset
            ep_from.append(starts)
            ep_to.append(ends)
            episode_source.append(np.full(len(starts), i, dtype=np.int64))
        self.episode_data_index = {"from": np.concatenate(ep_from), "to": np.concatenate(ep_to)}
        self.episode_dataset_index = np.concatenate(episode_source)
        self.dataset_index = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)

        if weights:
            w = np.asarray(weights, dtype=np.float64)
            if len(w) != len(self._datasets):
                raise ValueError(f"Got {len(w)} weights for {len(self._datasets)} datasets.")
            if np.any(w < 0) or w.sum() <= 0:
                raise ValueError(f"Dataset weights must be non-negative with a positive sum, got {weights}.")
            self.mixture_weights: np.ndarray | None = w / w.sum()
        else:
            self.mixture_weights = None

        self._frame_task: np.ndarray | None = None

    @property
    def frame_task(self) -> np.ndarray:
        """Per-frame task index, offset per source so ids are unique across the mixture."""
        if self._frame_task is None:
            parts, offset = [], 0
            for raw in self._raw_datasets:
                task = _lerobot_frame_task_index(raw)
                parts.append(task + offset)
                offset += int(task.max()) + 1 if len(task) else 0
            self._frame_task = np.concatenate(parts)
        return self._frame_task

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        if idx < 0:
            idx += self._length
        i = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        return self._datasets[i][idx - int(self._offsets[i])]


class MixtureSampler(torch.utils.data.Sampler[int]):
    """Draws frames from per-source groups with replacement: source i with probability weights[i].

    Sampling in two stages (pick a source, then a frame within it) keeps the draw exact without
    materializing a per-frame probability vector over the whole mixture. Order is always random —
    a weighted mixture has no meaningful sequential order.
    """

    def __init__(self, groups: Sequence[np.ndarray], weights: np.ndarray, num_samples: int, seed: int = 0):
        non_empty = [i for i, g in enumerate(groups) if len(g) > 0]
        if not non_empty:
            raise ValueError("MixtureSampler requires at least one non-empty group.")
        self._groups = [np.asarray(groups[i], dtype=np.int64) for i in non_empty]
        w = np.asarray(weights, dtype=np.float64)[non_empty]
        if w.sum() <= 0:
            raise ValueError("At least one non-empty group must have a positive weight.")
        self._weights = w / w.sum()
        self._num_samples = int(num_samples)
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng([self._seed, self._epoch])
        self._epoch += 1
        sources = rng.choice(len(self._groups), size=self._num_samples, p=self._weights)
        counts = np.bincount(sources, minlength=len(self._groups))
        picks = np.concatenate(
            [group[rng.integers(0, len(group), size=count)] for group, count in zip(self._groups, counts, strict=True)]
        )
        rng.shuffle(picks)
        return iter(picks.tolist())


def base_lerobot_dataset(dataset: Dataset) -> Dataset:
    """Strip the transform wrappers to reach the underlying LeRobot dataset or mixture."""
    while isinstance(dataset, TransformedDataset):
        dataset = dataset._dataset  # noqa: SLF001
    return dataset


def _lerobot_frame_task_index(dataset: lerobot_dataset.LeRobotDataset) -> np.ndarray:
    hf = dataset.hf_dataset
    try:
        # Arrow column access never touches the (lazily-decoded) image columns.
        return np.asarray(hf.data.column("task_index").to_numpy(zero_copy_only=False)).astype(np.int64)
    except Exception:  # Fall back to the datasets column accessor.
        return np.asarray(hf["task_index"]).astype(np.int64)


def per_frame_task_index(dataset: Dataset) -> np.ndarray:
    """Per-frame task index for a LeRobot dataset or a mixture of them."""
    if isinstance(dataset, MixedLeRobotDataset):
        return dataset.frame_task
    return _lerobot_frame_task_index(typing.cast(lerobot_dataset.LeRobotDataset, dataset))


def mixture_weights(dataset: Dataset) -> np.ndarray | None:
    """Sampling weights of a weighted mixture, or None for a plain (size-proportional) dataset."""
    return getattr(dataset, "mixture_weights", None)


def make_frame_sampler(
    base_dataset: Dataset,
    indices: np.ndarray,
    *,
    shuffle: bool,
    seed: int,
    num_samples: int | None = None,
) -> torch.utils.data.Sampler | list[int]:
    """Sampler over `indices`. Weighted mixtures draw per source; otherwise a plain shuffle."""
    indices = np.asarray(indices, dtype=np.int64)
    weights = mixture_weights(base_dataset)
    if weights is not None:
        source = base_dataset.dataset_index[indices]
        groups = [indices[source == i] for i in range(len(weights))]
        return MixtureSampler(groups, weights, num_samples or len(indices), seed=seed)
    if not shuffle:
        return indices.tolist()
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.SubsetRandomSampler(indices.tolist(), generator=generator)


def _split_episode_indices(
    dataset, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split a LeRobot dataset (or mixture) into train/val frame indices by episode.

    Mixtures are split per source, so every source contributes to both splits regardless of
    the size imbalance between them.

    Returns (train_indices, val_indices) as int64 arrays.
    """
    ep_from = np.asarray(dataset.episode_data_index["from"]).astype(np.int64)
    ep_to = np.asarray(dataset.episode_data_index["to"]).astype(np.int64)
    num_episodes = len(ep_from)

    episode_source = getattr(dataset, "episode_dataset_index", None)
    if episode_source is None:
        episode_source = np.zeros(num_episodes, dtype=np.int64)

    rng = np.random.RandomState(seed)
    val_eps: set[int] = set()
    for source in np.unique(episode_source):
        eps = np.flatnonzero(episode_source == source)
        perm = rng.permutation(len(eps))
        num_val = max(1, int(len(eps) * val_fraction))
        val_eps.update(eps[perm[:num_val]].tolist())

    train_indices = np.concatenate(
        [np.arange(ep_from[i], ep_to[i]) for i in range(num_episodes) if i not in val_eps]
    )
    val_indices = np.concatenate(
        [np.arange(ep_from[i], ep_to[i]) for i in range(num_episodes) if i in val_eps]
    )

    logging.info(
        f"Episode split: {num_episodes - len(val_eps)} train / {len(val_eps)} val episodes, "
        f"{len(train_indices)} train / {len(val_indices)} val frames"
    )
    return train_indices, val_indices


def _create_lerobot_source(
    repo_id: str, action_horizon: int, data_config: _config.DataConfig
) -> tuple[Dataset, lerobot_dataset.LeRobotDataset]:
    """Build one LeRobot dataset plus its prompt-resolved view."""
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    raw_dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    dataset: Dataset = raw_dataset
    if data_config.prompt_from_task:
        dataset = TransformedDataset(raw_dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset, raw_dataset


def create_lerobot_dataset(data_config: _config.DataConfig, action_horizon: int) -> Dataset:
    """Create the LeRobot dataset, mixing sources when `extra_repo_ids` is set."""
    if data_config.repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    repo_ids = [data_config.repo_id, *data_config.extra_repo_ids]
    if data_config.repo_weights and len(data_config.repo_weights) != len(repo_ids):
        raise ValueError(
            f"repo_weights has {len(data_config.repo_weights)} entries but there are {len(repo_ids)} repo ids."
        )
    sources = [_create_lerobot_source(repo_id, action_horizon, data_config) for repo_id in repo_ids]
    if len(sources) == 1:
        return sources[0][0]
    logging.info(
        f"Mixing {len(sources)} LeRobot datasets {repo_ids} "
        f"(weights: {list(data_config.repo_weights) or 'proportional to size'})"
    )
    return MixedLeRobotDataset(
        [dataset for dataset, _ in sources],
        [raw for _, raw in sources],
        data_config.repo_weights,
    )


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        if data_config.extra_repo_ids:
            raise ValueError("extra_repo_ids cannot be combined with the fake dataset.")
        return FakeDataset(model_config, num_samples=1024)

    return create_lerobot_dataset(data_config, action_horizon)


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
    tfds_split: str = "train",
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
        shuffle_buffer_size=data_config.rlds_shuffle_buffer_size,
        num_parallel_reads=data_config.rlds_num_parallel_reads,
        num_parallel_calls=data_config.rlds_num_parallel_calls,
        include_next_observation=data_config.include_next_observation,
        include_future_observation=data_config.include_future_observation,
        include_goal_observation=data_config.include_goal_observation,
        tfds_split=tfds_split,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        local_batch_size = config.batch_size // jax.process_count()
        logging.info(f"RLDS local_batch_size: {local_batch_size}")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=local_batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_train_val_data_loaders(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    framework: Literal["jax", "pytorch"] = "jax",
) -> tuple[DataLoader[tuple[_model.Observation, _model.Actions]], DataLoader[tuple[_model.Observation, _model.Actions]] | None]:
    """Create train and (optionally) val data loaders with episode-level splitting.

    Returns (train_loader, val_loader). val_loader is None when config.val_fraction == 0.
    """
    if config.val_fraction <= 0:
        train_loader = create_data_loader(config, sharding=sharding, shuffle=True, framework=framework)
        return train_loader, None

    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        local_batch_size = config.batch_size // jax.process_count()
        val_pct = int(config.val_fraction * 100)
        train_loader = create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=local_batch_size,
            sharding=sharding,
            shuffle=True,
            framework=framework,
            tfds_split=f"train[:{100 - val_pct}%]",
        )
        val_loader = create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=local_batch_size,
            sharding=sharding,
            shuffle=False,
            framework=framework,
            tfds_split=f"train[{100 - val_pct}%:]",
        )
        return train_loader, val_loader

    # LeRobot path: split by episode indices.
    dataset = create_torch_dataset(data_config, config.model.action_horizon, config.model)
    base_dataset = base_lerobot_dataset(dataset)
    train_indices, val_indices = _split_episode_indices(base_dataset, config.val_fraction, config.seed)

    dataset = transform_dataset(dataset, data_config)

    local_batch_size = config.batch_size // jax.process_count()
    train_sampler = make_frame_sampler(base_dataset, train_indices, shuffle=True, seed=config.seed)
    val_sampler = make_frame_sampler(base_dataset, val_indices, shuffle=True, seed=config.seed)

    train_torch_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        sampler=train_sampler,
        num_workers=config.num_workers,
        seed=config.seed,
        framework=framework,
    )
    val_torch_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        sampler=val_sampler,
        num_workers=0,
        seed=config.seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, train_torch_loader), DataLoaderImpl(data_config, val_torch_loader)


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    base_dataset = base_lerobot_dataset(dataset)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    if framework == "pytorch" and not torch.distributed.is_initialized():
        num_replicas, rank = 1, 0
        local_batch_size = batch_size
    else:
        if framework == "pytorch":
            num_replicas, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
        else:
            num_replicas, rank = jax.process_count(), jax.process_index()
        local_batch_size = batch_size // num_replicas

    sampler = None
    if mixture_weights(base_dataset) is not None:
        # Each replica draws from a disjoint stride of the frames, so the assembled global
        # batch still respects the mixture weights without duplicating frames across replicas.
        shard = np.arange(len(base_dataset), dtype=np.int64)[rank::num_replicas]
        sampler = make_frame_sampler(base_dataset, shard, shuffle=shuffle, seed=seed)
    elif num_replicas > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
    tfds_split: str = "train",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        tfds_split: The TFDS split string (e.g. "train", "train[:90%]").
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle, tfds_split=tfds_split)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            yielded_this_pass = False
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                yielded_this_pass = True
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)

            if not yielded_this_pass:
                raise RuntimeError(
                    "The DROID RLDS dataset produced no batches. This usually means the configured "
                    "filters (e.g. `filter_dict_path` or the success-trajectory filter) removed all "
                    "data, or the dataset/version in `datasets` doesn't match the data on disk."
                )


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
