import dataclasses

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


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

    for _, actions, future_state in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
        # Fake dataset never has future_state configured.
        assert future_state is None


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

    for _, actions, future_state in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
        assert future_state is None


# ---------------------------------------------------------------------------
# Unit tests for future-state sampling helpers
# ---------------------------------------------------------------------------


def test_sample_truncated_geometric_next_boundary():
    """With max_chunks=0 the sampler should return 0."""
    rng = np.random.default_rng(0)
    assert _data_loader._sample_truncated_geometric(0, 0.9, rng) == 0  # noqa: SLF001


def test_sample_truncated_geometric_range():
    """All samples must lie in [1, max_chunks]."""
    rng = np.random.default_rng(42)
    max_chunks = 5
    for _ in range(200):
        k = _data_loader._sample_truncated_geometric(max_chunks, 0.9, rng)  # noqa: SLF001
        assert 1 <= k <= max_chunks


def test_sample_truncated_geometric_decay_zero():
    """decay≈0 should always return 1 (concentrated at the nearest step)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = _data_loader._sample_truncated_geometric(10, 1e-10, rng)  # noqa: SLF001
        assert k == 1


def test_future_state_dataset_next():
    """FutureStateDataset with mode='next' returns state from action_horizon steps ahead."""
    action_horizon = 5
    state_dim = 4
    ep_size = 20  # single episode with 20 frames

    # Build a simple underlying dataset where state[i] = i * ones(state_dim).
    class _SimpleDataset(_data_loader.Dataset):
        def __getitem__(self, idx):
            return {"state": np.full(state_dim, float(idx), dtype=np.float32)}

        def __len__(self):
            return ep_size

    ep_from = np.array([0])
    ep_to = np.array([ep_size])
    frame_ep_idx = np.zeros(ep_size, dtype=np.int32)

    ds = _data_loader.FutureStateDataset(
        _SimpleDataset(),
        ep_from=ep_from,
        ep_to=ep_to,
        frame_episode_index=frame_ep_idx,
        action_horizon=action_horizon,
        config=_config.FutureStateConfig(mode="next"),
    )

    # For idx=3, future_idx = 3 + 5 = 8 → future_state should be all 8s.
    item = ds[3]
    assert item["future_state"][0] == pytest.approx(8.0)

    # For idx=17 (only 2 frames left, < action_horizon), max_chunks=0, reuse current.
    item = ds[17]
    assert item["future_state"][0] == pytest.approx(17.0)


def test_future_state_dataset_geometric_same_episode():
    """FutureStateDataset with mode='geometric' always stays within the episode."""
    action_horizon = 3
    ep_size = 30

    class _SimpleDataset(_data_loader.Dataset):
        def __getitem__(self, idx):
            return {"state": np.array([float(idx)], dtype=np.float32)}

        def __len__(self):
            return ep_size

    ep_from = np.array([0])
    ep_to = np.array([ep_size])
    frame_ep_idx = np.zeros(ep_size, dtype=np.int32)

    ds = _data_loader.FutureStateDataset(
        _SimpleDataset(),
        ep_from=ep_from,
        ep_to=ep_to,
        frame_episode_index=frame_ep_idx,
        action_horizon=action_horizon,
        config=_config.FutureStateConfig(mode="geometric", decay=0.9),
        seed=7,
    )

    for idx in range(ep_size):
        item = ds[idx]
        future_idx = int(item["future_state"][0])
        assert future_idx >= idx, "future_state must be at or after current frame"
        assert future_idx < ep_size, "future_state must stay within the episode"
        # Offset must be a multiple of action_horizon (or zero when at end).
        offset = future_idx - idx
        assert offset % action_horizon == 0, f"offset {offset} is not a multiple of action_horizon {action_horizon}"


def test_future_state_dataset_no_episode_crossing():
    """FutureStateDataset must not cross episode boundaries."""
    action_horizon = 4
    ep_size = 10

    class _TwoEpDataset(_data_loader.Dataset):
        def __getitem__(self, idx):
            return {"state": np.array([float(idx)])}

        def __len__(self):
            return ep_size * 2

    ep_from = np.array([0, ep_size])
    ep_to = np.array([ep_size, ep_size * 2])
    frame_ep_idx = np.array([0] * ep_size + [1] * ep_size, dtype=np.int32)

    ds = _data_loader.FutureStateDataset(
        _TwoEpDataset(),
        ep_from=ep_from,
        ep_to=ep_to,
        frame_episode_index=frame_ep_idx,
        action_horizon=action_horizon,
        config=_config.FutureStateConfig(mode="next"),
    )
    # ep 0: ep_end = ep_size - 2 = 8.
    # idx=8: max_chunks = (8-8)//4 = 0 → reuse current.
    assert ds[8]["future_state"][0] == pytest.approx(8.0)
    # idx=5: max_chunks = (8-5)//4 = 0 → reuse current (terminal frame excluded).
    assert ds[5]["future_state"][0] == pytest.approx(5.0)
    # idx=4: max_chunks = (8-4)//4 = 1 → future_idx = 4+4 = 8.
    assert ds[4]["future_state"][0] == pytest.approx(8.0)
    # ep 1: ep_end = ep_size*2 - 2 = 18.
    # idx=10: max_chunks = (18-10)//4 = 2 → future_idx = 10+4 = 14.
    assert ds[10]["future_state"][0] == pytest.approx(14.0)


def test_future_state_dataset_never_returns_last_frame():
    """future_state must never point to the terminal frame (ep_to - 1)."""
    action_horizon = 4
    ep_size = 12

    class _SimpleDataset(_data_loader.Dataset):
        def __getitem__(self, idx):
            return {"state": np.array([float(idx)])}

        def __len__(self):
            return ep_size

    ds = _data_loader.FutureStateDataset(
        _SimpleDataset(),
        ep_from=np.array([0]),
        ep_to=np.array([ep_size]),
        frame_episode_index=np.zeros(ep_size, dtype=np.int32),
        action_horizon=action_horizon,
        config=_config.FutureStateConfig(mode="next"),
    )
    last_frame_val = float(ep_size - 1)
    for idx in range(ep_size - 1):
        future_val = ds[idx]["future_state"][0]
        assert future_val != pytest.approx(last_frame_val), (
            f"idx={idx} returned terminal frame {ep_size - 1}"
        )


def test_future_state_geometric_decay_zero_returns_next():
    """Geometric mode with decay≈0 must return the next chunk, never current state."""
    action_horizon = 3
    ep_size = 20

    class _SimpleDataset(_data_loader.Dataset):
        def __getitem__(self, idx):
            return {"state": np.array([float(idx)])}

        def __len__(self):
            return ep_size

    ds = _data_loader.FutureStateDataset(
        _SimpleDataset(),
        ep_from=np.array([0]),
        ep_to=np.array([ep_size]),
        frame_episode_index=np.zeros(ep_size, dtype=np.int32),
        action_horizon=action_horizon,
        config=_config.FutureStateConfig(mode="geometric", decay=1e-10),
        seed=0,
    )
    for idx in range(ep_size):
        # ep_end = ep_size - 2; max_chunks = (ep_size - 2 - idx) // action_horizon
        max_chunks = (ep_size - 2 - idx) // action_horizon
        future_val = ds[idx]["future_state"][0]
        if max_chunks >= 1:
            assert future_val == pytest.approx(float(idx + action_horizon)), (
                f"idx={idx}: decay≈0 should give next chunk at {idx + action_horizon}, got {future_val}"
            )
        else:
            assert future_val == pytest.approx(float(idx))


def test_sample_truncated_geometric_decay_bias():
    """Lower decay yields a smaller mean sampled offset (concentrated near 1)."""
    rng_hi = np.random.default_rng(0)
    rng_lo = np.random.default_rng(0)
    n, max_k = 1000, 20
    mean_hi = np.mean([_data_loader._sample_truncated_geometric(max_k, 0.95, rng_hi) for _ in range(n)])  # noqa: SLF001
    mean_lo = np.mean([_data_loader._sample_truncated_geometric(max_k, 0.3, rng_lo) for _ in range(n)])  # noqa: SLF001
    assert mean_hi > mean_lo


def test_get_norm_stats_adds_future_state():
    """_get_norm_stats mirrors state stats to future_state when FutureStateConfig is set."""
    from openpi.shared.normalize import NormStats
    from openpi.training.config import DataConfig, FutureStateConfig

    stats = NormStats(mean=np.zeros(3), std=np.ones(3))
    base = DataConfig(repo_id="fake", norm_stats={"state": stats, "actions": stats})

    assert "future_state" not in _data_loader._get_norm_stats(base)  # noqa: SLF001

    with_fs = dataclasses.replace(base, future_state=FutureStateConfig())
    result = _data_loader._get_norm_stats(with_fs)  # noqa: SLF001
    assert result.get("future_state") is stats


def test_droid_inputs_passes_future_state():
    """DroidInputs transform should pass future_state through unchanged."""
    from openpi.models.model import ModelType
    from openpi.policies.droid_policy import DroidInputs

    transform = DroidInputs(model_type=ModelType.PI0)
    future = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    data = {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7),
        "observation/gripper_position": np.zeros(1),
        "prompt": "do something",
        "future_state": future,
    }
    out = transform(data)
    np.testing.assert_array_equal(out["future_state"], future)
