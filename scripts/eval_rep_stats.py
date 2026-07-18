"""Offline representation-statistics probe for any pi0 variant — the rep models (Pi0CRL /
Pi0SF / Pi0SP) and the baseline Pi0.

Loads a trained checkpoint, pulls a handful of val (or train) batches, extracts the phi/psi
representations of the current observation directly from ``model.get_state_representations`` and
prints a comprehensive set of *across-batch* collapse diagnostics pooled over all collected
samples — norms, per-feature std, effective rank, and off-diagonal cosine similarity. This is
the direct-from-reps complement to ``scripts/eval_rep.py`` (which aggregates each model's own
``compute_loss`` diagnostics instead).

"Across-batch" throughout means over the sample axis (axis 0), NOT within a sample's feature
vector: a rep that collapses to a batch-constant drives every std -> 0 while per-sample norms
stay large, so both are reported.

Rep extraction is uniform across variants: psi (and a prefix/"state" phi) come off one prefix
pass; whenever phi is action-dependent — the baseline Pi0's suffix-mean phi, or a rep with
``phi_input="state_action"`` — it is read at the batch's clean actions (t~=0) via
``get_phi_representation``, reusing the cached prefix. For the baseline the "reps" are the raw
mean prefix/suffix embeddings (the steering primitives), not learned readout heads.

The loader split matches training (episode-level, from config.seed / config.val_fraction), so
``--split val`` sees the same held-out episodes the run's val curve used. The checkpoint's
``params/`` holds the EMA weights when EMA was enabled.

Example:
  uv run scripts/eval_rep_stats.py \
      --config-name pi05_sp_libero_full_finetune_frozen_lite \
      --checkpoint-dir checkpoints/pi05_sp_libero_full_finetune_frozen_lite/my_exp/12000 \
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
import openpi.training.data_loader as _data_loader
import openpi.training.rl_data_loader as _rl

logger = logging.getLogger("openpi")


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
    # Cap the sample count used for the O(N^2) pairwise-cosine stat (0 = use all pooled).
    cosine_max_samples: int = 512
    # A feature counts as "dead" when its across-batch std falls below this.
    dead_std_threshold: float = 1e-3


def _as_batch_dict(batch) -> dict:
    """Normalize a loader batch to a dict. The goal-conditioned loader yields dicts; the
    standard loader yields (Observation, Actions) tuples."""
    if isinstance(batch, dict):
        return batch
    observation, actions = batch
    return {"observation": observation, "actions": actions}


def _rep_stats(rep: np.ndarray, cosine_max_samples: int, dead_std_threshold: float) -> dict[str, float]:
    """Across-sample collapse diagnostics for a pooled [N, D] rep (all float64)."""
    rep = np.asarray(rep, dtype=np.float64)
    n, d = rep.shape

    per_sample_norm = np.linalg.norm(rep, axis=-1)
    mean_rep = rep.mean(axis=0)  # [D]
    feat_std = rep.std(axis=0)  # [D], variation of each feature across the batch
    feat_var = feat_std**2
    # Participation ratio of the feature variances: an effective count of active dims,
    # in [1, D]. ~= D when variance is spread evenly, -> 1 as one direction dominates.
    eff_rank = float((feat_var.sum() ** 2) / (np.sum(feat_var**2) + 1e-12))

    # Mean off-diagonal cosine similarity: ~0 when reps point every which way, -> 1 under
    # directional collapse. Subsampled to bound the N^2 term.
    m = n if cosine_max_samples <= 0 else min(n, cosine_max_samples)
    sub = rep[:m]
    unit = sub / (np.linalg.norm(sub, axis=-1, keepdims=True) + 1e-12)
    cos = unit @ unit.T
    off = cos[~np.eye(m, dtype=bool)]
    mean_cos = float(off.mean()) if off.size else float("nan")

    return {
        "num_samples": float(n),
        "dim": float(d),
        "sample_norm_mean": float(per_sample_norm.mean()),
        "sample_norm_std": float(per_sample_norm.std()),
        "sample_norm_min": float(per_sample_norm.min()),
        "sample_norm_max": float(per_sample_norm.max()),
        # Norm of the across-batch mean rep. A large value with small feat_std = a constant
        # offset dominating the rep (a soft collapse the norm-only view would miss).
        "mean_norm": float(np.linalg.norm(mean_rep)),
        "feat_std_mean": float(feat_std.mean()),
        "feat_std_min": float(feat_std.min()),
        "feat_std_max": float(feat_std.max()),
        "dead_dim_frac": float(np.mean(feat_std < dead_std_threshold)),
        "eff_rank": eff_rank,
        "mean_offdiag_cos": mean_cos,
    }


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config = _config.get_config(args.config_name)
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.num_workers is not None:
        config = dataclasses.replace(config, num_workers=args.num_workers)

    params_path = pathlib.Path(args.checkpoint_dir) / "params"
    logger.info("Restoring params from %s", params_path)
    params = _model.restore_params(params_path)
    model = config.model.load(params)
    model.eval()

    if not hasattr(model, "get_state_representations"):
        raise ValueError(
            f"Model for config {args.config_name!r} has no get_state_representations; "
            "this script supports the pi0 variants (baseline Pi0, Pi0CRL, Pi0SF, Pi0SP)."
        )

    logger.info("Creating data loaders (split=%s)", args.split)
    if config.model.requires_goal_data:
        train_loader, val_loader = _rl.create_train_val_goal_conditioned_data_loaders(config)
    else:
        train_loader, val_loader = _data_loader.create_train_val_data_loaders(config)
    if args.split == "val":
        if val_loader is None:
            raise ValueError(f"Config {args.config_name} has val_fraction=0; no val split exists.")
        loader = val_loader
    elif args.split == "train":
        loader = train_loader
    else:
        raise ValueError(f"Unknown split: {args.split!r} (expected 'val' or 'train')")

    # phi always exists. psi is returned whenever the getter yields it — note Pi0SP with no
    # EMA target has psi_head=None yet mirrors phi into the psi slot, so we report what the
    # getter actually returned rather than the raw head presence.
    has_psi_head = getattr(model, "psi_head", None) is not None

    @nnx.jit
    def extract(model, batch):
        observation = batch["observation"]
        actions = batch["actions"]
        batch_size = actions.shape[0]
        # One prefix pass gives psi (and a "state" phi). Call obs-only so this works for the
        # baseline Pi0's obs-only signature too (the rep getter's actions/timestep default to
        # None). get_state_representations preprocesses the observation itself.
        psi, phi, kv_cache, prefix_mask, prefix_len = model.get_state_representations(observation)
        # phi is None when it is action-dependent (baseline suffix-mean phi, or a rep with
        # phi_input="state_action"): read it at the clean action (t~=0), reusing the prefix.
        if phi is None:
            timestep = jnp.full((batch_size,), 0.001)
            phi = model.get_phi_representation(
                observation, actions, timestep, kv_cache, prefix_mask, prefix_len
            )[0]
        out = {"phi": phi.astype(jnp.float32)}
        if psi is not None:
            out["psi"] = psi.astype(jnp.float32)
        return out

    collected: dict[str, list[np.ndarray]] = {}
    data_iter = iter(loader)
    for i in range(args.num_batches):
        batch = next(data_iter)
        reps = jax.device_get(extract(model, _as_batch_dict(batch)))
        for name, arr in reps.items():
            collected.setdefault(name, []).append(np.asarray(arr))
        logger.info("collected batch %d/%d (%s)", i + 1, args.num_batches, ", ".join(sorted(reps)))

    input_mode = {"phi": getattr(model, "phi_input", "?"), "psi": getattr(model, "psi_input", "?")}
    reps_note = " (psi mirrors phi: no separate psi head)" if "psi" in collected and not has_psi_head else ""
    print(
        f"\n=== {args.config_name} @ {args.checkpoint_dir} | split={args.split} "
        f"| {args.num_batches} batches x {config.batch_size} "
        f"| reps={'+'.join(sorted(collected))}{reps_note} ==="
    )
    for name in ("psi", "phi"):
        if name not in collected:
            continue
        pooled = np.concatenate(collected[name], axis=0)
        stats = _rep_stats(pooled, args.cosine_max_samples, args.dead_std_threshold)
        print(f"\n--- {name}  (input={input_mode[name]})  shape={pooled.shape} ---")
        for key in sorted(stats):
            print(f"{key:>20s}: {stats[key]:.5f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
