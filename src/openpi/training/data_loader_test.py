import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _StubLeRobotDataset:
    """Minimal stand-in exposing the LeRobot surface `MixedLeRobotDataset` reads."""

    def __init__(self, episode_lengths, task_ids, tag):
        ends = np.cumsum(episode_lengths)
        self.episode_data_index = {"from": np.concatenate([[0], ends[:-1]]), "to": ends}
        self.task_ids = np.asarray(task_ids)
        self._tag = tag

    @property
    def hf_dataset(self):
        raise AssertionError("frame_task should read the stubbed task column")

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, index):
        return {"tag": self._tag, "local_index": int(index)}


def _make_mixture(weights=()):
    a = _StubLeRobotDataset([3, 3], [0, 0, 0, 1, 1, 1], "a")
    b = _StubLeRobotDataset([2, 2], [0, 0, 1, 1], "b")
    return _data_loader.MixedLeRobotDataset([a, b], [a, b], weights)


def test_mixed_dataset_global_index(monkeypatch):
    monkeypatch.setattr(_data_loader, "_lerobot_frame_task_index", lambda ds: ds.task_ids)
    mixed = _make_mixture()

    assert len(mixed) == 10
    assert mixed[0] == {"tag": "a", "local_index": 0}
    assert mixed[5] == {"tag": "a", "local_index": 5}
    assert mixed[6] == {"tag": "b", "local_index": 0}
    assert mixed[9] == {"tag": "b", "local_index": 3}

    np.testing.assert_array_equal(mixed.episode_data_index["from"], [0, 3, 6, 8])
    np.testing.assert_array_equal(mixed.episode_data_index["to"], [3, 6, 8, 10])
    np.testing.assert_array_equal(mixed.episode_dataset_index, [0, 0, 1, 1])
    np.testing.assert_array_equal(mixed.dataset_index, [0] * 6 + [1] * 4)
    # Task ids are offset per source so the two sources never collide.
    np.testing.assert_array_equal(mixed.frame_task, [0, 0, 0, 1, 1, 1, 2, 2, 3, 3])


def test_mixture_weights_drive_sampling(monkeypatch):
    monkeypatch.setattr(_data_loader, "_lerobot_frame_task_index", lambda ds: ds.task_ids)
    # Source "b" holds 4 of 10 frames but should supply ~90% of the draws.
    mixed = _make_mixture(weights=(0.1, 0.9))
    sampler = _data_loader.make_frame_sampler(mixed, np.arange(len(mixed)), shuffle=True, seed=0, num_samples=20000)

    drawn = np.asarray(list(sampler))
    assert len(drawn) == 20000
    from_b = (mixed.dataset_index[drawn] == 1).mean()
    assert 0.88 < from_b < 0.92
    # Every frame of each source stays reachable.
    assert set(np.unique(drawn)) == set(range(10))


def test_mixture_sampler_reshuffles_each_epoch(monkeypatch):
    monkeypatch.setattr(_data_loader, "_lerobot_frame_task_index", lambda ds: ds.task_ids)
    mixed = _make_mixture(weights=(1.0, 1.0))
    sampler = _data_loader.make_frame_sampler(mixed, np.arange(len(mixed)), shuffle=True, seed=0)

    assert list(sampler) != list(sampler)


def test_unweighted_mixture_uses_plain_shuffle(monkeypatch):
    monkeypatch.setattr(_data_loader, "_lerobot_frame_task_index", lambda ds: ds.task_ids)
    mixed = _make_mixture()

    assert _data_loader.mixture_weights(mixed) is None
    sampler = _data_loader.make_frame_sampler(mixed, np.arange(len(mixed)), shuffle=True, seed=0)
    # A plain shuffle is a permutation: every frame exactly once, so sampling is size-proportional.
    assert sorted(sampler) == list(range(10))


def test_split_is_stratified_across_sources(monkeypatch):
    monkeypatch.setattr(_data_loader, "_lerobot_frame_task_index", lambda ds: ds.task_ids)
    mixed = _make_mixture()

    train_indices, val_indices = _data_loader._split_episode_indices(  # noqa: SLF001
        mixed, val_fraction=0.5, seed=0
    )

    # Both sources contribute one episode to each split.
    for indices in (train_indices, val_indices):
        assert set(np.unique(mixed.dataset_index[indices])) == {0, 1}
    assert set(train_indices) | set(val_indices) == set(range(10))
    assert not set(train_indices) & set(val_indices)


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
