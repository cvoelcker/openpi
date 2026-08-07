"""Pooled spectral-rank probe for pi0 representations.

Two phases, so the expensive one runs once:

1. **extract** (GPU): pull `--num-batches` batches, compute every requested representation,
   and append each to a float32 `.npy` on disk. Pooling across batches decouples the sample
   count from the GPU batch size, which matters because every rank statistic here is capped
   at min(N, D) — a single 512-sample batch cannot express a rank above 512.
2. **analyze** (CPU, no JAX): memory-map the cache and accumulate the D x D covariance in
   chunks, so peak memory is O(D^2) regardless of N. Rerunning with `--reuse-cache` skips
   phase 1 entirely.

Representations, all read off the SAME batches so the comparison is paired:

- ``prefix_mean`` — `Pi0.get_prefix_mean_embedding`, the mean-pooled backbone prefix. This is
  the steering baseline's rep, and it exists for every variant.
- ``phi`` / ``psi`` — the learned readout heads, when the model has them.

Because the rep variants freeze the backbone (`action_loss_coeff=0` and
`rep_backbone_grad_scale=0` => `backbone_frozen`) and warm-start it from the pi05 checkpoint,
``prefix_mean`` taken off a rep checkpoint is bit-for-bit the baseline rep it was warm-started
from. So a single rep checkpoint yields both arms of the baseline-vs-learned-head comparison on
identical data — no second model, no second data loader, no distribution shift.

Each rep is analyzed twice: raw, and L2-normalized per sample. Consumers that score cosine
similarity (or set `normalize_reps`) see the latter, and per-sample normalization is a nonlinear
map that can move the spectrum, so the raw number alone can mislead.

`--reps prefix_mean` costs one prefix pass per batch; adding `phi`/`psi` costs a second, since
the public rep accessors run their own prefix pass.

Example:
  uv run scripts/probe_rep_rank.py \
      --config-name pi05_sp_td_libero_mixed_norm_big \
      --checkpoint-dir checkpoints/pi05_sp_td_libero_mixed_norm_big/td_big_fixed_.../40000 \
      --num-batches 100 --split train --cache-dir /tmp/rank_probe_40000
"""

import dataclasses
import json
import logging
import pathlib

import numpy as np
import tyro

logger = logging.getLogger("openpi")

# Rep names this script knows how to extract.
_ALL_REPS = ("prefix_mean", "phi", "psi")


@dataclasses.dataclass
class Args:
    config_name: str
    # Checkpoint step directory (must contain a params/ subdirectory). Not needed with
    # --reuse-cache, which reads the stored reps and never builds the model.
    checkpoint_dir: str | None = None
    cache_dir: str = "rep_rank_cache"
    num_batches: int = 100
    # "train" or "val" (the held-out episodes, same split as training via config.seed).
    split: str = "train"
    # Comma-separated subset of prefix_mean,phi,psi. Heads absent from the model are skipped.
    reps: str = "prefix_mean,phi,psi"
    # Skip extraction and analyze whatever is already in cache_dir.
    reuse_cache: bool = False
    # Optional loader overrides; None keeps the config's values. Do NOT override config.seed —
    # it determines the episode split.
    batch_size: int | None = None
    num_workers: int | None = None
    # Refuse to extract if the cache would exceed this. Bytes = N * D * 4 per rep.
    max_cache_gb: float = 8.0
    # Cap the samples used in the analysis (0 = all). The covariance is exact for any N; this
    # only bounds runtime.
    max_analysis_samples: int = 0
    # Rows per chunk when accumulating the covariance. Peak analysis memory is roughly
    # chunk_rows * D * 8 bytes plus the D x D matrix.
    chunk_rows: int = 8192


# --------------------------------------------------------------------------------------
# Phase 2: analysis. Pure numpy, no JAX — safe to run anywhere.
# --------------------------------------------------------------------------------------


def _covariance(path: pathlib.Path, *, normalize: bool, chunk_rows: int, max_samples: int):
    """Two chunked passes over a memory-mapped [N, D] float32 cache -> (cov, n, d).

    Never materializes more than `chunk_rows` rows in float64, so peak memory is
    O(chunk_rows * D + D^2) whatever N is.
    """
    data = np.load(path, mmap_mode="r")
    n_total, d = data.shape
    n = n_total if max_samples <= 0 else min(n_total, max_samples)

    def _chunk(start: int) -> np.ndarray:
        block = np.asarray(data[start : min(start + chunk_rows, n)], dtype=np.float64)
        if normalize:
            block /= np.linalg.norm(block, axis=-1, keepdims=True) + 1e-12
        return block

    mean = np.zeros(d, dtype=np.float64)
    for start in range(0, n, chunk_rows):
        mean += _chunk(start).sum(axis=0)
    mean /= n

    # Scatter of the centered data. Centering inside the chunk loop keeps this exact rather
    # than relying on the E[xx^T] - mu mu^T identity, which cancels catastrophically when the
    # mean offset dominates the variance — exactly the regime these reps sit in.
    scatter = np.zeros((d, d), dtype=np.float64)
    for start in range(0, n, chunk_rows):
        centered = _chunk(start) - mean
        scatter += centered.T @ centered
    return scatter / n, n, d


def _rank_metrics(cov: np.ndarray, n: int, d: int) -> dict[str, float]:
    """Spectral rank statistics of a covariance matrix.

    `participation_ratio` is the same quantity `rep_base.batch_rep_stats` logs online as
    `{name}_eff_rank`, so the two are directly comparable — but it is a variance-weighted
    softening of rank, not a rank: a single dominant eigenvalue drives it toward 1 no matter
    how many directions are nonzero. The other statistics disagree with it in informative
    ways, so read them together:

    - `entropy_eff_rank` — exp(Shannon entropy of the normalized spectrum), the Roy-Vetterli
      effective rank. Far less dominated by lambda_max than the participation ratio, so a
      large gap between the two means "one big direction plus a long tail" rather than
      "genuinely k-dimensional".
    - `stable_rank` — trace / lambda_max. The most pessimistic of the three.
    - `dim_90/95/99` — how many leading directions it takes to hold that share of the total
      variance. The most interpretable answer to "how many dimensions is this really".
    - `numerical_rank_*` — eigenvalue counts above a relative tolerance; these are the only
      entries that speak to rank in the linear-algebra sense.
    """
    eig = np.linalg.eigvalsh(cov)[::-1]
    eig = np.clip(eig, 0.0, None)
    total = float(eig.sum())
    if total <= 0.0:
        return {"num_samples": float(n), "dim": float(d), "total_variance": 0.0}

    p = eig / total
    nonzero = p[p > 0]
    cumulative = np.cumsum(p)
    ceiling = min(n - 1, d)  # centering costs one degree of freedom

    out = {
        "num_samples": float(n),
        "dim": float(d),
        "rank_ceiling": float(ceiling),
        "total_variance": total,
        "participation_ratio": total**2 / float(np.sum(eig**2)),
        "entropy_eff_rank": float(np.exp(-np.sum(nonzero * np.log(nonzero)))),
        "stable_rank": total / float(eig[0]),
        "top1_variance_frac": float(p[0]),
        "top10_variance_frac": float(cumulative[min(9, d - 1)]),
    }
    for share in (0.90, 0.95, 0.99):
        out[f"dim_{int(share * 100)}"] = float(np.searchsorted(cumulative, share) + 1)
    for tol in (1e-2, 1e-4, 1e-6):
        out[f"numerical_rank_{tol:.0e}"] = float(np.sum(eig > tol * eig[0]))
    # Dimension-free versions, so reps of different width stay comparable.
    out["participation_ratio_frac"] = out["participation_ratio"] / ceiling
    out["entropy_eff_rank_frac"] = out["entropy_eff_rank"] / ceiling
    out["dim_95_frac"] = out["dim_95"] / ceiling
    return out


def analyze(cache_dir: pathlib.Path, args: Args) -> None:
    caches = sorted(cache_dir.glob("*.npy"))
    if not caches:
        raise FileNotFoundError(f"No cached reps (*.npy) in {cache_dir}")

    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    print(f"\n=== rank probe | {cache_dir} ===")
    if meta:
        print(f"    config={meta.get('config_name')} checkpoint={meta.get('checkpoint_dir')}")
        print(f"    split={meta.get('split')} batches={meta.get('num_batches')} batch_size={meta.get('batch_size')}")

    rows: dict[str, dict[str, float]] = {}
    for path in caches:
        for normalize in (False, True):
            cov, n, d = _covariance(
                path,
                normalize=normalize,
                chunk_rows=args.chunk_rows,
                max_samples=args.max_analysis_samples,
            )
            label = f"{path.stem}{'  [L2]' if normalize else '  [raw]'}"
            rows[label] = _rank_metrics(cov, n, d)

    keys = [
        "num_samples",
        "dim",
        "rank_ceiling",
        "participation_ratio",
        "entropy_eff_rank",
        "stable_rank",
        "dim_90",
        "dim_95",
        "dim_99",
        "numerical_rank_1e-02",
        "numerical_rank_1e-04",
        "numerical_rank_1e-06",
        "top1_variance_frac",
        "top10_variance_frac",
        "participation_ratio_frac",
        "entropy_eff_rank_frac",
        "dim_95_frac",
    ]
    width = max(len(label) for label in rows)
    header = "".join(f"{label:>22s}" for label in rows)
    print(f"\n{'metric':<{width}s}{header}")
    for key in keys:
        line = f"{key:<{width}s}"
        for metrics in rows.values():
            value = metrics.get(key)
            line += f"{'-':>22s}" if value is None else f"{value:>22.4f}"
        print(line)
    print(
        "\nparticipation_ratio matches rep_base.batch_rep_stats's *_eff_rank. It is variance-"
        "weighted:\na large gap against entropy_eff_rank / dim_95 means one dominant direction "
        "plus a long tail,\nnot a genuinely low-dimensional rep. rank_ceiling = min(N-1, D) "
        "bounds every column."
    )


# --------------------------------------------------------------------------------------
# Phase 1: extraction. Imports JAX lazily so `--reuse-cache` works without it.
# --------------------------------------------------------------------------------------


def extract(cache_dir: pathlib.Path, args: Args) -> None:
    # Imported here, not at module scope, so `--reuse-cache` analyzes a cache on a machine
    # without JAX installed (jax[cuda12] has no macOS wheels).
    import flax.nnx as nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    import openpi.models.model as _model  # noqa: PLC0415
    import openpi.training.config as _config  # noqa: PLC0415
    import openpi.training.data_loader as _data_loader  # noqa: PLC0415
    import openpi.training.rl_data_loader as _rl  # noqa: PLC0415

    if args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required unless --reuse-cache is set")

    requested = [name.strip() for name in args.reps.split(",") if name.strip()]
    if unknown := set(requested) - set(_ALL_REPS):
        raise ValueError(f"Unknown reps {sorted(unknown)}; choose from {_ALL_REPS}")

    config = _config.get_config(args.config_name)
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.num_workers is not None:
        config = dataclasses.replace(config, num_workers=args.num_workers)

    params_path = pathlib.Path(args.checkpoint_dir) / "params"
    logger.info("Restoring params from %s", params_path)
    model = config.model.load(_model.restore_params(params_path))
    model.eval()

    want_heads = [name for name in requested if name in ("phi", "psi")]
    has_heads = hasattr(model, "get_state_representations") and getattr(model, "phi_head", None) is not None
    if want_heads and not has_heads:
        logger.warning("Model has no phi/psi heads; extracting %s only", [n for n in requested if n not in want_heads])
        want_heads = []
    want_prefix_mean = "prefix_mean" in requested

    @nnx.jit
    def compute(model, observation, actions):
        out = {}
        if want_prefix_mean:
            out["prefix_mean"] = model.get_prefix_mean_embedding(observation)[0].astype(jnp.float32)
        if want_heads:
            psi, phi, kv_cache, prefix_mask, prefix_len = model.get_state_representations(observation)
            if phi is None:
                # An action-dependent phi (suffix-mean baseline, or phi_input="state_action")
                # is read at the clean action, t ~= 0, reusing the cached prefix.
                timestep = jnp.full((actions.shape[0],), 0.001)
                phi = model.get_phi_representation(observation, actions, timestep, kv_cache, prefix_mask, prefix_len)[0]
            if "phi" in want_heads and phi is not None:
                out["phi"] = phi.astype(jnp.float32)
            if "psi" in want_heads and psi is not None:
                out["psi"] = psi.astype(jnp.float32)
        return out

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

    cache_dir.mkdir(parents=True, exist_ok=True)
    collected: dict[str, list[np.ndarray]] = {}
    data_iter = iter(loader)
    for i in range(args.num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            logger.warning("Loader exhausted after %d batches", i)
            break
        if isinstance(batch, dict):
            observation, actions = batch["observation"], batch["actions"]
        else:
            observation, actions = batch
        for name, arr in jax.device_get(compute(model, observation, actions)).items():
            collected.setdefault(name, []).append(np.asarray(arr, dtype=np.float32))

        if i == 0:
            # Size the run before it fills a disk: every batch contributes the same bytes.
            per_batch = sum(arr[0].nbytes for arr in collected.values())
            total_gb = per_batch * args.num_batches / 1e9
            logger.info(
                "Reps per batch: %s | projected cache %.2f GB",
                {name: tuple(arr[0].shape) for name, arr in collected.items()},
                total_gb,
            )
            if total_gb > args.max_cache_gb:
                raise ValueError(
                    f"Projected cache {total_gb:.2f} GB exceeds --max-cache-gb {args.max_cache_gb}. "
                    f"Lower --num-batches / --batch-size, or raise the cap."
                )
        if (i + 1) % 10 == 0:
            logger.info("batch %d/%d", i + 1, args.num_batches)

    for name, chunks in collected.items():
        pooled = np.concatenate(chunks, axis=0)
        np.save(cache_dir / f"{name}.npy", pooled)
        logger.info("wrote %s %s", cache_dir / f"{name}.npy", pooled.shape)

    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "config_name": args.config_name,
                "checkpoint_dir": args.checkpoint_dir,
                "split": args.split,
                "num_batches": args.num_batches,
                "batch_size": config.batch_size,
                "reps": sorted(collected),
            },
            indent=2,
        )
    )


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cache_dir = pathlib.Path(args.cache_dir)
    if not args.reuse_cache:
        extract(cache_dir, args)
    analyze(cache_dir, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
