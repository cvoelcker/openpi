"""Dump per-epoch batch-composition statistics of the training frame sampler.

Replays the real sampler built by the training data loaders (no dataset __getitem__, so no
image decode) and reports, for every epoch, how each batch is composed: source mix, how many
distinct frames and episodes it contains, and how many of its frames are temporal neighbours.

Usage:
    uv run scripts/diagnose_mixture_sampler.py pi05_sp_libero_recommended_mixed --num-epochs 17
"""

import dataclasses

import numpy as np
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.rl_data_loader as _rl_data_loader


@dataclasses.dataclass
class Args:
    config_name: tyro.conf.Positional[str]
    num_epochs: int = 17
    split: str = "train"
    # Overrides the config's batch size; pass whatever the run used.
    batch_size: int | None = None
    # Batches per epoch to score; None uses every batch.
    max_batches: int | None = None
    # Two frames of the same episode this far apart or closer count as near-duplicates.
    neighbor_window: int | None = None


def _batch_stats(
    picks: np.ndarray,
    batch_size: int,
    dataset_index: np.ndarray,
    ep_start: np.ndarray,
    neighbor_window: int,
    max_batches: int | None,
) -> dict[str, float]:
    num_batches = len(picks) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    batches = picks[: num_batches * batch_size].reshape(num_batches, batch_size)

    # Frames of an episode are contiguous in the global index, so sorting by frame index also
    # groups by episode and puts temporal neighbours next to each other.
    ordered = np.sort(batches, axis=1)
    step = np.diff(ordered, axis=1)
    episodes = ep_start[ordered]
    same_episode = episodes[:, 1:] == episodes[:, :-1]

    return {
        "frac_src0": float(np.mean(dataset_index[batches] == 0)),
        "uniq_frames": float(np.mean(1 + np.count_nonzero(step, axis=1))),
        "uniq_episodes": float(np.mean(1 + np.count_nonzero(np.diff(episodes, axis=1), axis=1))),
        "neighbor_pairs": float(np.mean(np.count_nonzero((step <= neighbor_window) & same_episode, axis=1))),
        "frame_mean": float(np.mean(batches)),
        "frame_std": float(np.mean(np.std(batches, axis=1))),
        "num_batches": num_batches,
    }


def main(args: Args) -> None:
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    neighbor_window = args.neighbor_window or config.model.action_horizon

    dataset = _data_loader.create_lerobot_dataset(data_config, config.model.action_horizon)
    source_dataset = _data_loader.base_lerobot_dataset(dataset)

    train_indices, val_indices = _data_loader._split_episode_indices(  # noqa: SLF001
        source_dataset, config.val_fraction, config.seed
    )
    indices = train_indices if args.split == "train" else val_indices

    dataset_index = getattr(source_dataset, "dataset_index", np.zeros(len(source_dataset), dtype=np.int64))
    ep_start, _ = _rl_data_loader._per_frame_episode_bounds(source_dataset)  # noqa: SLF001
    group_sizes = np.bincount(dataset_index[indices], minlength=int(dataset_index.max()) + 1)

    sampler = _rl_data_loader._process_sharded_sampler(  # noqa: SLF001
        source_dataset, indices, shuffle=True, seed=config.seed
    )
    batch_size = args.batch_size or config.batch_size

    print(f"config={args.config_name} split={args.split} batch_size={batch_size} seed={config.seed}")
    print(f"sampler={type(sampler).__name__} frames={len(indices)} per-source={group_sizes.tolist()}")
    print(f"steps/epoch={len(indices) / batch_size:.1f} neighbor_window={neighbor_window}\n")

    header = f"{'epoch':>5} {'frac_src0':>10} {'uniq_frames':>12} {'uniq_eps':>9} {'nbr_pairs':>10} {'frame_std':>11} {'epoch_cov':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for epoch in range(args.num_epochs):
        picks = np.fromiter(iter(sampler), dtype=np.int64)
        stats = _batch_stats(picks, batch_size, dataset_index, ep_start, neighbor_window, args.max_batches)
        # Fraction of the split's frames the epoch touches at least once.
        coverage = len(np.unique(picks)) / len(indices)
        rows.append(stats)
        print(
            f"{epoch:5d} {stats['frac_src0']:10.4f} {stats['uniq_frames']:12.2f} {stats['uniq_episodes']:9.2f} "
            f"{stats['neighbor_pairs']:10.3f} {stats['frame_std']:11.1f} {coverage:10.4f}"
        )

    print("\nz-scores across epochs (|z| > 3 means that epoch's batches are composed differently):")
    for key in ("frac_src0", "uniq_frames", "uniq_episodes", "neighbor_pairs", "frame_std"):
        values = np.array([r[key] for r in rows])
        spread = values.std()
        z = (values - values.mean()) / spread if spread > 0 else np.zeros_like(values)
        flagged = [f"{i}:{zi:+.1f}" for i, zi in enumerate(z) if abs(zi) > 3]
        print(f"  {key:<15} mean={values.mean():12.4f} std={spread:9.4f}  outliers: {', '.join(flagged) or 'none'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
