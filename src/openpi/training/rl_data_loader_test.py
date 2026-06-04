import dataclasses
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest

import openpi.models.model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import rl_data_loader as _rl

STATE_DIM = 4
ACTION_HORIZON = 3
ACTION_DIM = 4
IMG_SHAPE = (8, 8, 3)


class _FakeTransformedDataset(_data_loader.Dataset):
    """Mimics the dict produced by openpi's transform pipeline for a pi0 model.

    Every field is filled with the frame index so we can assert which frame was fetched.
    """

    def __init__(self, num_frames: int):
        self._num_frames = num_frames

    def __getitem__(self, index) -> dict:
        idx = int(index)
        return {
            "image": {"base_0_rgb": np.full(IMG_SHAPE, idx % 256, dtype=np.uint8)},
            "image_mask": {"base_0_rgb": np.True_},
            "state": np.full((STATE_DIM,), idx, dtype=np.float32),
            "tokenized_prompt": np.zeros((6,), dtype=np.int32),
            "tokenized_prompt_mask": np.ones((6,), dtype=bool),
            "actions": np.full((ACTION_HORIZON, ACTION_DIM), idx, dtype=np.float32),
        }

    def __len__(self) -> int:
        return self._num_frames


def _bounds_from_episodes(episodes: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    num_frames = max(end for _, end in episodes)
    ep_start = np.zeros(num_frames, dtype=np.int64)
    ep_end = np.zeros(num_frames, dtype=np.int64)
    for start, end in episodes:
        ep_start[start:end] = start
        ep_end[start:end] = end
    return ep_start, ep_end


def _make_dataset(
    episodes, *, gamma=0.9, future_uniform=True, seed=0, action_chunk_size=1
) -> _rl.RandomFutureDataset:
    ep_start, ep_end = _bounds_from_episodes(episodes)
    num_frames = len(ep_start)
    return _rl.RandomFutureDataset(
        _FakeTransformedDataset(num_frames),
        ep_start,
        ep_end,
        sampling=_rl.GoalSamplingConfig(gamma=gamma, future_uniform=future_uniform, seed=seed),
        action_chunk_size=action_chunk_size,
    )


def test_per_frame_episode_bounds():
    class _Meta:
        episode_data_index = {"from": np.array([0, 5]), "to": np.array([5, 9])}

        def __len__(self):
            return 9

    ep_start, ep_end = _rl._per_frame_episode_bounds(_Meta())  # noqa: SLF001
    assert ep_start.tolist() == [0, 0, 0, 0, 0, 5, 5, 5, 5]
    assert ep_end.tolist() == [5, 5, 5, 5, 5, 9, 9, 9, 9]


@pytest.mark.parametrize("future_uniform", [True, False])
def test_sampling_invariants(future_uniform):
    episodes = [(0, 5), (5, 9)]
    ep_start, ep_end = _bounds_from_episodes(episodes)
    ds = _make_dataset(episodes, future_uniform=future_uniform)

    for t in range(len(ep_start)):
        start, end = int(ep_start[t]), int(ep_end[t])
        for _ in range(300):
            next_idx, future_idx, goal_idx, pads = ds._sample_indices(t)  # noqa: SLF001

            # All indices must stay within the same episode.
            assert start <= next_idx < end
            assert start <= future_idx < end
            assert start <= goal_idx < end

            if t == end - 1:
                # Last frame of episode: everything clamps to t and is flagged as padding.
                assert next_idx == future_idx == goal_idx == t
                assert bool(pads["next_is_pad"])
                assert bool(pads["future_is_pad"])
                assert bool(pads["goal_is_pad"])
            else:
                assert next_idx == t + 1
                assert not bool(pads["next_is_pad"])
                assert not bool(pads["future_is_pad"])
                assert not bool(pads["goal_is_pad"])
                # "future-then-goal": t < future <= goal
                assert t < future_idx <= goal_idx


def test_goal_offset_clamped_to_episode():
    # With gamma close to 1, the geometric draw is huge and must clamp to the episode end.
    episodes = [(0, 6)]
    ds = _make_dataset(episodes, gamma=0.999999)
    for _ in range(50):
        _, future_idx, goal_idx, _ = ds._sample_indices(0)  # noqa: SLF001
        assert goal_idx == 5  # last frame of the single episode
        assert 0 < future_idx <= 5


@pytest.mark.parametrize("chunk_size", [2, 3])
def test_chunk_indices_are_chunk_aligned(chunk_size):
    """All sampled indices must be multiples of chunk_size ahead of t."""
    # Episode long enough to have several complete chunks.
    episodes = [(0, 20)]
    ds = _make_dataset(episodes, gamma=0.9, action_chunk_size=chunk_size)
    for t in range(20):
        for _ in range(100):
            next_idx, future_idx, goal_idx, pads = ds._sample_indices(t)  # noqa: SLF001
            if not pads["next_is_pad"]:
                assert (next_idx - t) % chunk_size == 0, f"next not chunk-aligned: t={t}, next={next_idx}"
            if not pads["future_is_pad"]:
                assert (future_idx - t) % chunk_size == 0
                assert (goal_idx - t) % chunk_size == 0
                assert future_idx <= goal_idx


@pytest.mark.parametrize("chunk_size", [2, 3])
def test_chunk_last_chunk_frames_are_padded(chunk_size):
    """Frames where fewer than chunk_size steps remain must have all indices padded."""
    # Episode of length 10: frames 0-9. With chunk_size=3, frames 7,8,9 have <3 frames remaining.
    episodes = [(0, 10)]
    ds = _make_dataset(episodes, gamma=0.9, action_chunk_size=chunk_size)
    for t in range(10):
        frames_remaining = 10 - 1 - t  # frames strictly after t
        _, _, _, pads = ds._sample_indices(t)  # noqa: SLF001
        if frames_remaining < chunk_size:
            assert pads["next_is_pad"], f"t={t} should have next_is_pad=True"
            assert pads["future_is_pad"]
            assert pads["goal_is_pad"]
        else:
            assert not pads["future_is_pad"], f"t={t} should have future available"
            assert not pads["goal_is_pad"]


def test_sampling_is_seeded_deterministic():
    episodes = [(0, 10)]
    ds_a = _make_dataset(episodes, seed=123)
    ds_b = _make_dataset(episodes, seed=123)
    seq_a = [ds_a._sample_indices(0) for _ in range(20)]  # noqa: SLF001
    seq_b = [ds_b._sample_indices(0) for _ in range(20)]  # noqa: SLF001
    assert [(n, f, g) for n, f, g, _ in seq_a] == [(n, f, g) for n, f, g, _ in seq_b]


def test_item_structure_strips_aux_actions():
    ds = _make_dataset([(0, 5)])
    item = ds[0]
    assert set(item) == {
        "observation",
        "next_observation",
        "future_observation",
        "goal_observation",
        "next_is_pad",
        "future_is_pad",
        "goal_is_pad",
        "actions",
    }
    # Auxiliary frames should not carry the (redundant) action chunk.
    for key in ["observation", "next_observation", "future_observation", "goal_observation"]:
        assert "actions" not in item[key]
    assert item["actions"].shape == (ACTION_HORIZON, ACTION_DIM)


def test_anchor_observation_matches_index():
    ds = _make_dataset([(0, 5)])
    item = ds[2]
    # The anchor observation/actions must correspond to frame t=2.
    assert np.all(item["observation"]["state"] == 2)
    assert np.all(item["actions"] == 2)


def test_end_to_end_collation_and_observation():
    episodes = [(0, 6), (6, 12)]
    ds = _make_dataset(episodes)
    loader = _data_loader.TorchDataLoader(ds, local_batch_size=4, num_batches=2, num_workers=0, seed=0)
    gc_loader = _rl.GoalConditionedDataLoader(data_config=None, data_loader=loader)

    batches = list(gc_loader)
    assert len(batches) == 2

    for batch in batches:
        for name in ["observation", "next_observation", "future_observation", "goal_observation"]:
            obs = batch[name]
            assert isinstance(obs, _model.Observation)
            assert obs.state.shape == (4, STATE_DIM)
            assert obs.images["base_0_rgb"].shape == (4, *IMG_SHAPE)
        assert batch["actions"].shape == (4, ACTION_HORIZON, ACTION_DIM)
        for mask in ["next_is_pad", "future_is_pad", "goal_is_pad"]:
            assert np.asarray(batch[mask]).shape == (4,)


# Repo id of a converted LeRobot dataset to use for the manual integration test.
# Override with the env var if your dataset lives under a different repo id.
_MANUAL_REPO_ID = os.environ.get("OPENPI_RL_TEST_REPO_ID", "LIBERO/lerobot_format")


@pytest.mark.manual
def test_create_goal_conditioned_data_loader_real_dataset():
    """End-to-end test against a real LeRobot dataset.

    This is gated behind `-m manual` because it requires a converted dataset on disk
    (set HF_LEROBOT_HOME / HF_HOME accordingly), network access for the tokenizer, and
    triggers a one-time HuggingFace split generation. It uses `num_workers >= 1` so that
    tokenization runs in spawned worker subprocesses (running it in the main process can
    crash due to a TensorFlow/sentencepiece ABI conflict).
    """
    # INSERT_YOUR_CODE
    hf_cache = os.environ.get("HF_HOME") or os.environ.get("HF_LEROBOT_HOME")
    print(f"hf_cache is: {hf_cache!r}, set: {bool(hf_cache)}")
    batch_size = 4
    config = _config.get_config("pi0_libero")
    config = dataclasses.replace(config, batch_size=batch_size, num_workers=2)
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, repo_id=_MANUAL_REPO_ID))

    loader = _rl.create_goal_conditioned_data_loader(
        config,
        sampling=_rl.GoalSamplingConfig(gamma=0.95, seed=0),
        shuffle=True,
        num_batches=1,
        skip_norm_stats=True,
    )

    assert loader.data_config().repo_id == _MANUAL_REPO_ID

    batch = next(iter(loader))
    
    import pdb; pdb.set_trace()

    action_horizon = config.model.action_horizon
    action_dim = config.model.action_dim
    for name in ["observation", "next_observation", "future_observation", "goal_observation"]:
        obs = batch[name]
        assert isinstance(obs, _model.Observation)
        assert obs.state.shape == (batch_size, action_dim)
        for image in obs.images.values():
            assert image.shape == (batch_size, 224, 224, 3)
    assert batch["actions"].shape == (batch_size, action_horizon, action_dim)
    for mask in ["next_is_pad", "future_is_pad", "goal_is_pad"]:
        assert np.asarray(batch[mask]).shape == (batch_size,)
