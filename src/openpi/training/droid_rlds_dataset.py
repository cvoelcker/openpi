"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from collections.abc import Sequence
import dataclasses
from enum import Enum
from enum import auto
import json
import logging

import os
import time
from contextlib import contextmanager

import tqdm

import openpi.shared.download as download

# Enable extra TF-side debug mapping when this env var is set.
DEBUG_RLDS = os.environ.get("OPENPI_RLDS_DEBUG") == "1"


@contextmanager
def log_stage(name: str):
    t0 = time.time()
    logging.info("[RLDS DEBUG] Stage start: %s", name)
    try:
        yield
    finally:
        logging.info("[RLDS DEBUG] Stage done: %s (%.3fs)", name, time.time() - t0)


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


def build_filter_table(filter_dict_path: str, tf):
    """Build (or load from a disk cache) the StaticHashTable of allowed frame keys.

    The filter dict maps each episode key to ranges of frames to keep. Expanding
    those ranges into one key per kept frame produces tens of millions of entries
    for full DROID; doing it in a Python loop is slow and allocates a large list
    of Python ``str`` objects every run. We expand once, serialize the resulting
    string tensor next to the downloaded JSON, and reload it on subsequent runs.

    The cache file name embeds the JSON's size and mtime, so a re-downloaded /
    changed filter dict transparently invalidates the cache.
    """
    cached_json_path = download.maybe_download(filter_dict_path)
    stat = cached_json_path.stat()
    cache_path = cached_json_path.with_name(
        f"{cached_json_path.name}.keys-{stat.st_size}-{stat.st_mtime_ns}.tftensor"
    )

    if cache_path.exists():
        with log_stage("load_filter_keys_cache"):
            keys_tensor = tf.io.parse_tensor(tf.io.read_file(str(cache_path)), out_type=tf.string)
    else:
        with log_stage("build_filter_keys"):
            with cached_json_path.open("r") as f:
                filter_dict = json.load(f)
            logging.info(f"Building filter keys for {len(filter_dict)} episodes")
            keys: list[str] = []
            for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Expanding filter ranges..."):
                for start, end in ranges:
                    keys.extend(f"{episode_key}--{t}" for t in range(start, end))
            keys_tensor = tf.constant(keys, dtype=tf.string)
            del keys
        with log_stage("write_filter_keys_cache"):
            # Write atomically so an interrupted run never leaves a corrupt cache.
            tmp_path = cache_path.with_name(cache_path.name + ".tmp")
            tf.io.write_file(str(tmp_path), tf.io.serialize_tensor(keys_tensor))
            os.replace(tmp_path, cache_path)

    num_keys = int(tf.shape(keys_tensor)[0])
    logging.info(f"Filter table contains {num_keys} frame keys")
    # Values are uniformly True; reconstruct them rather than storing a second tensor.
    values_tensor = tf.ones([num_keys], dtype=tf.bool)
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
    )


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        # HER (Hindsight Experience Replay) goal-conditioned sampling.
        # When set, each frame will additionally carry next_observation, future_observation,
        # and goal_observation sampled from the same trajectory using a geometric distribution
        # with discount factor her_gamma. Set to None to disable.
        her_gamma: float | None = None,
        # Which auxiliary HER observations to actually gather and decode. Each disabled
        # observation removes one full (image) copy per frame -- this reduces gather work,
        # image decodes, and shuffle-buffer memory. Ignored when her_gamma is None.
        include_next_observation: bool = True,
        include_future_observation: bool = True,
        include_goal_observation: bool = True,
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        # Ensure dataset weights sum to 1.0
        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        def prepare_single_dataset(dataset_cfg: RLDSDataset):
            # ds_name, version = dataset_name.split(":")
            ds_name, version = dataset_cfg.name, dataset_cfg.version
            builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
            dataset = dl.DLataset.from_rlds(
                builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )

            # Filter out any unsuccessful trajectories -- we use the file name to check this
            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                )
            )

            # Repeat dataset so we never run out of data.
            dataset = dataset.repeat()

            # Load the filter dictionary if provided.
            # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
            # (e.g.,
            # {
            #     "<episode key>": [[0, 100], [200, 300]]
            # }
            # means keep frames 0-99 and 200-299).

            filter_dict_path = dataset_cfg.filter_dict_path
            if filter_dict_path is not None:
                self.filter_table = build_filter_table(filter_dict_path, tf)
                logging.info("Filter hash table initialized")
            else:
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
                )

            def restructure(traj):
                """Reformat observation and action keys, sample language instruction."""
                # Important: we use joint *position* action space -- easier to simulate!
                actions = tf.concat(
                    (
                        (
                            traj["action_dict"]["joint_position"]
                            if action_space == DroidActionSpace.JOINT_POSITION
                            else traj["action_dict"]["joint_velocity"]
                        ),
                        traj["action_dict"]["gripper_position"],
                    ),
                    axis=-1,
                )
                # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
                # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
                exterior_img = tf.cond(
                    tf.random.uniform(shape=[]) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )
                wrist_img = traj["observation"]["wrist_image_left"]
                # Randomly sample one of the three language instructions
                instruction = tf.random.shuffle(
                    [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
                )[0]

                traj_len = tf.shape(traj["action"])[0]
                indices = tf.as_string(tf.range(traj_len))

                # Data filtering:
                # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
                # and each step's time step index. This will index into the filter hash table, and if it returns true,
                # then the frame passes the filter.
                step_id = (
                    traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                    + "--"
                    + traj["traj_metadata"]["episode_metadata"]["file_path"]
                    + "--"
                    + indices
                )
                passes_filter = self.filter_table.lookup(step_id)

                return {
                    "actions": actions,
                    "observation": {
                        "image": exterior_img,
                        "wrist_image": wrist_img,
                        "joint_position": traj["observation"]["joint_position"],
                        "gripper_position": traj["observation"]["gripper_position"],
                    },
                    "prompt": instruction,
                    "step_id": step_id,
                    "passes_filter": passes_filter,
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):
                """Splits episode into action chunks."""
                traj_len = tf.shape(traj["actions"])[0]

                # For each step in the trajectory, construct indices for the next n actions
                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )

                # Cap to length of the sequence --> final chunks will repeat the last action
                # This makes sense, since we are using absolute joint + gripper position actions
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                # Gather the actions for each chunk
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # HER sampling: for each timestep t, gather next/future/goal observations from
            # the same trajectory before flattening (so future frames are still accessible).
            if her_gamma is not None:
                her_gamma_val = float(her_gamma)

                def add_her_samples(traj):
                    traj_len = tf.shape(traj["actions"])[0]
                    indices = tf.cast(tf.range(traj_len), tf.int32)
                    C = action_chunk_size  # treat each chunk as one atomic step

                    # next: frame after executing one full action chunk
                    next_indices = tf.minimum(indices + C, traj_len - 1)
                    next_is_pad = (indices + C) >= traj_len

                    # number of full chunks reachable from each step
                    num_chunks = (traj_len - 1 - indices) // C  # [T], in chunk units
                    has_future = num_chunks > 0

                    # goal: truncated geometric(p=1-gamma) in chunk units
                    p = float(max(1e-6, 1.0 - her_gamma_val))
                    u = tf.random.uniform([traj_len], dtype=tf.float64, minval=1e-10, maxval=1.0)
                    log_1mp = tf.cast(tf.math.log(tf.constant(1.0 - p, dtype=tf.float64)), tf.float64)
                    raw_goal_chunks = tf.cast(tf.math.ceil(tf.math.log(u) / log_1mp), tf.int32)
                    raw_goal_chunks = tf.maximum(raw_goal_chunks, 1)
                    goal_offset_chunks = tf.where(
                        has_future, tf.minimum(raw_goal_chunks, num_chunks), tf.ones_like(indices)
                    )
                    goal_indices = indices + goal_offset_chunks * C

                    # future: uniform in [1, goal_offset_chunks] chunks
                    fu = tf.random.uniform([traj_len], dtype=tf.float32)
                    future_offset_chunks = tf.maximum(
                        tf.cast(fu * tf.cast(goal_offset_chunks, tf.float32), tf.int32), 1
                    )
                    future_offset_chunks = tf.minimum(future_offset_chunks, goal_offset_chunks)
                    future_indices = indices + future_offset_chunks * C

                    # for last-chunk frames clamp everything to t and mark as padded
                    goal_indices = tf.where(has_future, goal_indices, indices)
                    future_indices = tf.where(has_future, future_indices, indices)

                    def gather_obs(obs, idx):
                        return tf.nest.map_structure(lambda x: tf.gather(x, idx), obs)

                    if include_next_observation:
                        traj["next_observation"] = gather_obs(traj["observation"], next_indices)
                        traj["next_is_pad"] = next_is_pad
                    if include_future_observation:
                        traj["future_observation"] = gather_obs(traj["observation"], future_indices)
                        traj["future_is_pad"] = tf.logical_not(has_future)
                    if include_goal_observation:
                        traj["goal_observation"] = gather_obs(traj["observation"], goal_indices)
                        traj["goal_is_pad"] = tf.logical_not(has_future)
                    return traj

                dataset = dataset.traj_map(add_her_samples, num_parallel_calls)

            # Flatten: map from trajectory dataset to dataset of individual action chunks
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            # Filter data that doesn't pass the filter
            def filter_from_dict(frame):
                return frame["passes_filter"]

            dataset = dataset.filter(filter_from_dict)

            # Remove "passes_filter" key from output
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            dataset = dataset.map(remove_passes_filter)

            # Decode images: RLDS saves encoded images, only decode now for efficiency.
            # When HER is enabled, also decode images in the auxiliary observation dicts.
            def decode_images(frame):
                def decode_obs(obs):
                    obs["image"] = tf.io.decode_image(obs["image"], expand_animations=False, dtype=tf.uint8)
                    obs["wrist_image"] = tf.io.decode_image(
                        obs["wrist_image"], expand_animations=False, dtype=tf.uint8
                    )
                    return obs

                # Some versions of the TF/DL pipeline may pass a non-dict (tuple/tensor)
                # to this function during tracing. Try dict-style access first and
                # fall back to tuple-style indexing if that fails.
                try:
                    frame_ob = frame["observation"]
                except Exception:
                    # Fall back: assume observation is first element
                    try:
                        frame_ob = frame[0]
                    except Exception:
                        frame_ob = frame

                frame_ob = decode_obs(frame_ob)

                # Reassign back into frame. If frame supports string keys, use them,
                # otherwise try tuple/list assignment where possible.
                try:
                    frame["observation"] = frame_ob
                except Exception:
                    try:
                        # convert to list to allow assignment if it's a tuple
                        f_list = list(frame)
                        f_list[0] = frame_ob
                        frame = tuple(f_list)
                    except Exception:
                        # give up and return the decoded observation as-is
                        return frame_ob

                if her_gamma is not None:
                    try:
                        if include_next_observation:
                            frame["next_observation"] = decode_obs(frame["next_observation"])
                        if include_future_observation:
                            frame["future_observation"] = decode_obs(frame["future_observation"])
                        if include_goal_observation:
                            frame["goal_observation"] = decode_obs(frame["goal_observation"])
                    except Exception:
                        pass
                return frame

            return dataset.frame_map(decode_images, num_parallel_calls)

        logging.info(f"Preparing {len(datasets)} datasets...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)

        with log_stage("prepare_all_datasets"):
            all_datasets = [prepare_single_dataset(dataset) for dataset in datasets]
        weights = [dataset.weight for dataset in datasets]

        with log_stage("finalize_dataset"):
            final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
            final_dataset = final_dataset.shuffle(shuffle_buffer_size)
            final_dataset = final_dataset.batch(batch_size)
            # Note =>> Seems to reduce memory usage without affecting speed?
            # final_dataset = final_dataset.with_ram_budget(1)

            self.dataset = final_dataset
            self.batch_size = batch_size
            self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000
