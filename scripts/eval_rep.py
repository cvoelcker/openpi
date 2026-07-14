"""Offline rep-checkpoint evaluation: load a trained pi05rep checkpoint and push goal-conditioned
batches through `compute_loss` (train=False), aggregating all diagnostics — rep_loss, the
masked/nomask collision decomposition, and the val ranking probe (rank_order_*, rank_neg_*,
ranking_loss). No training state, no wandb; results print to stdout.

The loader split is identical to training (episode-level, derived from config.seed and
config.val_fraction), so `--split val` evaluates the same held-out episodes the run's val curve
used. The checkpoint's `params/` holds the EMA params when EMA was enabled — the same weights
`val_step` evaluates.

Example:
  uv run scripts/eval_rep.py \
      --config-name pi05_crl_libero_full_finetune_frozen_lite \
      --checkpoint-dir checkpoints/pi05_crl_libero_full_finetune_frozen_lite/my_exp/12000 \
      --num-batches 20 --split val
"""

import dataclasses
import logging
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.rl_data_loader as _rl

logger = logging.getLogger("openpi")

_RANKING_KEYS = (
    "frame_index",
    "next_frame_index",
    "next_is_pad",
    "future_frame_index",
    "future_is_pad",
    "future_episode_id",
    "goal_frame_index",
    "goal_is_pad",
    "negative_frame_index",
    "negative_episode_id",
)


@dataclasses.dataclass
class Args:
    config_name: str
    # Checkpoint step directory (must contain a params/ subdirectory).
    checkpoint_dir: str
    num_batches: int = 20
    # "val" (held-out episodes, same split as training) or "train".
    split: str = "val"
    # Optional overrides; None keeps the config's values. Do NOT override config.seed — it
    # determines the episode split, and changing it would leak train episodes into "val".
    batch_size: int | None = None
    num_workers: int | None = None
    # RNG seed for preprocessing/noise inside compute_loss (not the data split).
    eval_seed: int = 0


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = _config.get_config(args.config_name)
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.num_workers is not None:
        config = dataclasses.replace(config, num_workers=args.num_workers)

    params_path = pathlib.Path(args.checkpoint_dir) / "params"
    logger.info(f"Restoring params from {params_path}")
    params = _model.restore_params(params_path)
    model = config.model.load(params)
    model.eval()

    logger.info("Creating data loaders (split=%s)", args.split)
    train_loader, val_loader = _rl.create_train_val_goal_conditioned_data_loaders(config)
    if args.split == "val":
        if val_loader is None:
            raise ValueError(f"Config {args.config_name} has val_fraction=0; no val split exists.")
        loader = val_loader
    elif args.split == "train":
        loader = train_loader
    else:
        raise ValueError(f"Unknown split: {args.split!r} (expected 'val' or 'train')")

    @nnx.jit
    def eval_step(model, rng, observation, future_observation, actions, episode_id,
                  negative_observation, next_observation, goal_observation, ranking_indices):
        chunked_loss, log_dict = model.compute_loss(
            rng,
            observation,
            future_observation,
            actions,
            train=False,
            episode_id=episode_id,
            negative_observation=negative_observation,
            next_observation=next_observation,
            goal_observation=goal_observation,
            ranking_indices=ranking_indices,
        )
        return {"loss": jnp.mean(chunked_loss), **{k: jnp.mean(v) for k, v in log_dict.items()}}

    rng = jax.random.key(args.eval_seed)
    results: list[dict] = []
    data_iter = iter(loader)
    for i in range(args.num_batches):
        batch = next(data_iter)
        episode_id = batch.get("episode_id")
        # Ranking probe needs the frame indices; skip gracefully for stale loaders without them.
        do_ranking = "frame_index" in batch and "future_episode_id" in batch and episode_id is not None
        ranking_indices = {k: batch[k] for k in _RANKING_KEYS if k in batch} if do_ranking else None
        info = eval_step(
            model,
            jax.random.fold_in(rng, i),
            batch["observation"],
            batch["future_observation"],
            batch["actions"],
            episode_id,
            batch.get("negative_observation"),
            batch.get("next_observation") if do_ranking else None,
            batch.get("goal_observation") if do_ranking else None,
            ranking_indices,
        )
        info = jax.device_get(info)
        results.append(info)
        logger.info(
            "batch %d/%d: rep_loss=%.4f%s",
            i + 1,
            args.num_batches,
            info.get("rep_loss", float("nan")),
            f" rank_order_acc={info['rank_order_acc']:.3f}" if "rank_order_acc" in info else "",
        )

    print(f"\n=== {args.config_name} @ {args.checkpoint_dir} | split={args.split} "
          f"| {args.num_batches} batches x {config.batch_size} ===")
    for key in sorted(results[0]):
        values = np.array([r[key] for r in results], dtype=np.float64)
        print(f"{key:>24s}: mean={values.mean():.4f}  std={values.std():.4f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
