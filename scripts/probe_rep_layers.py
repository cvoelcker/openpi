"""Layer-wise linear probe for CRL representation quality across backbone depth.

Motivation
----------
In the CRL rep-head redesign we plan to feed backbone hidden states into a small
dedicated transformer that outputs phi/psi. The open question is *which* backbone
features to read: the final-layer output (``prefix_out``) is the most optimized for
action prediction and may have compressed away the goal/temporal structure the
contrastive objective needs. Middle layers often carry richer, less task-specialized
information (the standard probing-literature finding).

This script answers that empirically and cheaply, *before* we commit to any tapping
plumbing. For a frozen, trained checkpoint it:

1. runs the (frozen) backbone over ``observation`` (current) and ``future_observation``,
2. extracts a pooled feature per backbone depth. Depths ``0..L-1`` use the per-layer
   key projections the transformer stack already returns (a compressed proxy for the
   residual stream *entering* each layer -- requires no changes to ``gemma.py``); the
   ``final(res)`` point uses the true final-layer residual output ``prefix_out``,
3. fits a *linear* symmetric-InfoNCE probe (phi = W_phi @ h_current, psi = W_psi @
   h_future) per depth on cached train features,
4. reports train/val InfoNCE and top-1 in-batch accuracy per depth.

If the metric peaks in the middle and is much worse at ``final(res)``, that quantifies
how action-specialized the top of the backbone is, and tells us which layer the rep
head should read. If final ~= middle, we just read the final layer and skip the tap.

NOTE: layers ``0..L-1`` are in key-projection space (dim = num_kv_heads * head_dim),
while ``final(res)`` is in residual space (dim = model width). Absolute InfoNCE is not
directly comparable across those two spaces; the *shape of the trend across depth* is
the signal, and ``final(res)`` is the honest apples-to-apples reference for "read the
final residual". The per-layer key trend tells you where information rises/falls.

Usage (run on a GPU box / cluster, not a laptop -- this loads a multi-B backbone):

    uv run scripts/probe_rep_layers.py pi05_libero_rep \
        --checkpoint /path/to/checkpoints/.../<step>/params \
        --batch-size 128 --train-batches 40 --val-batches 40 --out probe.png

Smoke-test the probe-fitting logic locally (no backbone, no checkpoint):

    uv run scripts/probe_rep_layers.py --smoke-test
"""

import argparse
import dataclasses
import logging
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.model as _model
from openpi.models.pi05rep import make_attn_mask
import openpi.training.config as _config
import openpi.training.rl_data_loader as _rl_data_loader
import openpi.training.sharding as sharding

logger = logging.getLogger("probe")


# --------------------------------------------------------------------------------------
# Linear InfoNCE probe (the part worth unit-testing; independent of the backbone).
# --------------------------------------------------------------------------------------
def _infonce(phi: jnp.ndarray, psi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Symmetric in-batch InfoNCE (softmax-CE form) + top-1 accuracy.

    phi, psi: (n, rep_dim). Positives are the diagonal of phi @ psi.T.
    Mirrors the contrast in pi05rep.compute_loss (raw dot products, no temperature),
    using the standard cross-entropy form so the value is a clean bits-style metric.
    """
    logits = phi @ psi.T  # (n, n): [i, j] = <phi_i, psi_j>
    labels = jnp.arange(logits.shape[0])
    ce = optax.softmax_cross_entropy_with_integer_labels
    loss = 0.5 * (ce(logits, labels).mean() + ce(logits.T, labels).mean())
    acc = (jnp.argmax(logits, axis=1) == labels).mean()
    return loss, acc


def fit_probe(
    train_c: np.ndarray,
    train_f: np.ndarray,
    val_c: np.ndarray,
    val_f: np.ndarray,
    *,
    rep_dim: int,
    steps: int,
    batch: int,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Fit a linear phi/psi map with InfoNCE on train features; eval train+val.

    train/val features are (current, future) pairs: arrays of shape (N, feat_dim).
    Train and val are evaluated with the *same* minibatch size so the InfoNCE scale
    (which grows with the number of in-batch negatives) is comparable across splits.
    """
    train_c, train_f = jnp.asarray(train_c), jnp.asarray(train_f)
    val_c, val_f = jnp.asarray(val_c), jnp.asarray(val_f)
    d = train_c.shape[1]
    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    params = {
        "Wc": 0.02 * jax.random.normal(k1, (d, rep_dim)),
        "Wf": 0.02 * jax.random.normal(k2, (d, rep_dim)),
    }
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(params, c, f):
        phi = c @ params["Wc"]
        psi = f @ params["Wf"]
        loss, acc = _infonce(phi, psi)
        return loss, acc

    # Whole training loop in one jitted scan: minibatch sampling, gather, and the
    # optimizer step all run on-device, so there's no per-step Python/dispatch overhead.
    @jax.jit
    def train_loop(params, opt_state, rng):
        def step(carry, step_rng):
            params, opt_state = carry
            idx = jax.random.choice(step_rng, train_c.shape[0], (batch,), replace=False)
            (_loss, _acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, train_c[idx], train_f[idx]
            )
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        (params, opt_state), _ = jax.lax.scan(step, (params, opt_state), jax.random.split(rng, steps))
        return params, opt_state

    @jax.jit
    def eval_metrics(params, c, f):
        # Average InfoNCE/accuracy over consecutive minibatches of `batch`.
        n = (c.shape[0] // batch) * batch
        c, f = c[:n].reshape(-1, batch, d), f[:n].reshape(-1, batch, d)

        def body(carry, xs):
            cc, ff = xs
            loss, acc = loss_fn(params, cc, ff)
            return carry, (loss, acc)

        _, (losses, accs) = jax.lax.scan(body, None, (c, f))
        return losses.mean(), accs.mean()

    params, opt_state = train_loop(params, opt_state, key)
    tr_loss, tr_acc = eval_metrics(params, train_c, train_f)
    va_loss, va_acc = eval_metrics(params, val_c, val_f)
    return {
        "train_loss": float(tr_loss),
        "val_loss": float(va_loss),
        "train_acc": float(tr_acc),
        "val_acc": float(va_acc),
        "gap": float(va_loss - tr_loss),
    }


# --------------------------------------------------------------------------------------
# Backbone feature extraction.
# --------------------------------------------------------------------------------------
def build_feature_fn(graphdef):
    """Return a jitted fn (state, obs, future) -> (per_layer_keys, final_residual).

    per_layer_keys: (L, 2B, key_dim) pooled key vectors, current=[:, :B], future=[:, B:].
    final_residual: (2B, width) pooled final-layer residual output.
    """

    @jax.jit
    def feature_fn(state, obs, future):
        model = nnx.merge(graphdef, state)
        model.eval()
        key = jax.random.key(0)  # unused when train=False (no augmentation)
        obs = _model.preprocess_observation(key, obs, train=False)
        future = _model.preprocess_observation(key, future, train=False)
        stacked = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), obs, future)

        tokens, mask, ar_mask = model.embed_prefix(stacked)
        attn_mask = make_attn_mask(mask, ar_mask)
        positions = jnp.cumsum(mask, axis=1) - 1
        # Prefix-only forward: second (action) expert is not run. Mirrors the call in
        # pi05rep.get_state_representations (adarms_cond defaults to None per expert).
        outs, kv_cache = model.PaliGemma.llm([tokens, None], mask=attn_mask, positions=positions)
        prefix_out = outs[0]  # (2B, T, width)
        keys = kv_cache[0]  # (L, 2B, T, num_kv_heads, head_dim)
        L = keys.shape[0]
        keys = keys.reshape(L, keys.shape[1], keys.shape[2], -1)  # (L, 2B, T, key_dim)

        # Pool over the real state tokens (images + language); drop the trailing
        # phi/psi rep tokens so we measure backbone content, not the rep readouts.
        num_rep = model._num_prefix_rep_tokens
        valid = mask
        if num_rep > 0:
            valid = valid.at[:, mask.shape[1] - num_rep :].set(False)
        w = valid.astype(jnp.float32)  # (2B, T)
        denom = jnp.clip(w.sum(axis=1), 1.0)  # (2B,)

        per_layer = (keys * w[None, :, :, None]).sum(axis=2) / denom[None, :, None]  # (L, 2B, key_dim)
        final_res = (prefix_out * w[:, :, None]).sum(axis=1) / denom[:, None]  # (2B, width)
        return per_layer, final_res

    return feature_fn


def cache_features(feature_fn, state, loader, num_batches: int):
    """Run `num_batches` batches through the backbone; return cached numpy features.

    Returns (per_layer_c, per_layer_f, final_c, final_f):
      per_layer_*: (L, N, key_dim); final_*: (N, width).
    """
    layer_c, layer_f, fin_c, fin_f = [], [], [], []
    logger.info("fetching first batch (spawning data workers; may take a while)...")
    it = iter(loader)
    for b in range(num_batches):
        t0 = time.time()
        batch = next(it)
        if b == 0:
            logger.info(
                "first batch fetched in %.1fs; running first forward "
                "(JIT-compiling the backbone, can take several minutes)...",
                time.time() - t0,
            )
        tf = time.time()
        per_layer, final_res = feature_fn(state, batch["observation"], batch["future_observation"])
        per_layer = np.asarray(jax.device_get(per_layer))
        final_res = np.asarray(jax.device_get(final_res))
        bsz = per_layer.shape[1] // 2
        layer_c.append(per_layer[:, :bsz])
        layer_f.append(per_layer[:, bsz:])
        fin_c.append(final_res[:bsz])
        fin_f.append(final_res[bsz:])
        # Log every batch for the first few (so the first compile is visible), then every 10.
        if b < 3 or (b + 1) % 10 == 0:
            logger.info("cached %d/%d batches (forward %.1fs)", b + 1, num_batches, time.time() - tf)
    return (
        np.concatenate(layer_c, axis=1),
        np.concatenate(layer_f, axis=1),
        np.concatenate(fin_c, axis=0),
        np.concatenate(fin_f, axis=0),
    )


# --------------------------------------------------------------------------------------
# Reporting.
# --------------------------------------------------------------------------------------
def report(rows: list[tuple[str, dict[str, float]]], out_path: str | None):
    header = f"{'depth':>12} | {'train_loss':>10} {'val_loss':>10} {'gap':>8} | {'train_acc':>9} {'val_acc':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name, m in rows:
        print(
            f"{name:>12} | {m['train_loss']:>10.4f} {m['val_loss']:>10.4f} {m['gap']:>8.4f} | "
            f"{m['train_acc']:>9.4f} {m['val_acc']:>9.4f}"
        )
    if out_path is None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return
    names = [n for n, _ in rows]
    x = list(range(len(names)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(x, [m["train_loss"] for _, m in rows], "-o", label="train")
    ax1.plot(x, [m["val_loss"] for _, m in rows], "-o", label="val")
    ax1.set_title("Linear-probe InfoNCE vs depth")
    ax1.set_ylabel("InfoNCE")
    ax2.plot(x, [m["train_acc"] for _, m in rows], "-o", label="train")
    ax2.plot(x, [m["val_acc"] for _, m in rows], "-o", label="val")
    ax2.set_title("In-batch top-1 accuracy vs depth")
    ax2.set_ylabel("accuracy")
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_xlabel("backbone depth")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    logger.info("wrote plot to %s", out_path)


def run_probe(args):
    config = _config.get_config(args.config_name)
    if args.batch_size is not None:
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if config.val_fraction <= 0:
        raise ValueError("Probe needs a val split; use a config with val_fraction > 0.")

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    logger.info("building model...")
    model = config.model.create(jax.random.key(0))
    logger.info("restoring checkpoint params from %s (this can take a while)...", args.checkpoint)
    # Preserve the checkpoint's dtypes; the LLM casts activations to its embed_dtype
    # internally, so mixed fp32/bf16 params are fine for a frozen forward.
    params = _model.restore_params(args.checkpoint)
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    logger.info("params restored; building data loaders (loads the LeRobot dataset)...")

    train_loader, val_loader = _rl_data_loader.create_train_val_goal_conditioned_data_loaders(
        config, sharding=data_sharding
    )
    if val_loader is None:
        raise ValueError("No val loader was created.")
    logger.info("data loaders ready.")

    feature_fn = build_feature_fn(graphdef)
    logger.info("caching train features (%d batches)", args.train_batches)
    tr_layer_c, tr_layer_f, tr_fin_c, tr_fin_f = cache_features(feature_fn, state, train_loader, args.train_batches)
    logger.info("caching val features (%d batches)", args.val_batches)
    va_layer_c, va_layer_f, va_fin_c, va_fin_f = cache_features(feature_fn, state, val_loader, args.val_batches)

    num_layers = tr_layer_c.shape[0]
    rows: list[tuple[str, dict[str, float]]] = []
    for layer in range(num_layers):
        m = fit_probe(
            tr_layer_c[layer].astype(np.float32),
            tr_layer_f[layer].astype(np.float32),
            va_layer_c[layer].astype(np.float32),
            va_layer_f[layer].astype(np.float32),
            rep_dim=args.rep_dim,
            steps=args.probe_steps,
            batch=args.probe_batch,
        )
        rows.append((f"L{layer}(k)", m))
        logger.info("layer %d: val_loss=%.4f val_acc=%.4f gap=%.4f", layer, m["val_loss"], m["val_acc"], m["gap"])
    m = fit_probe(
        tr_fin_c.astype(np.float32),
        tr_fin_f.astype(np.float32),
        va_fin_c.astype(np.float32),
        va_fin_f.astype(np.float32),
        rep_dim=args.rep_dim,
        steps=args.probe_steps,
        batch=args.probe_batch,
    )
    rows.append(("final(res)", m))
    report(rows, args.out)


def run_smoke_test(args):
    """Validate the probe-fitting logic on synthetic features (no backbone)."""
    rng = np.random.default_rng(0)
    n, latent, feat, rep = 4000, 8, 64, 16
    # Structured case: current/future share a latent (should be learnable -> high acc).
    z = rng.standard_normal((n, latent))
    proj_c, proj_f = rng.standard_normal((latent, feat)), rng.standard_normal((latent, feat))
    tr_c = z @ proj_c + 0.3 * rng.standard_normal((n, feat))
    tr_f = z @ proj_f + 0.3 * rng.standard_normal((n, feat))
    zv = rng.standard_normal((n, latent))
    va_c = zv @ proj_c + 0.3 * rng.standard_normal((n, feat))
    va_f = zv @ proj_f + 0.3 * rng.standard_normal((n, feat))
    m = fit_probe(tr_c, tr_f, va_c, va_f, rep_dim=rep, steps=args.probe_steps, batch=args.probe_batch)
    print("structured pair (expect val_acc >> 1/batch):", m)
    # Random case: no shared structure (should be ~1/batch chance accuracy).
    rc = rng.standard_normal((n, feat))
    rf = rng.standard_normal((n, feat))
    m2 = fit_probe(rc, rf, rng.standard_normal((n, feat)), rng.standard_normal((n, feat)),
                   rep_dim=rep, steps=args.probe_steps, batch=args.probe_batch)
    print(f"random pair (expect val_acc ~ 1/{args.probe_batch} = {1/args.probe_batch:.4f}):", m2)


def main():
    # force=True resets any handlers already installed by absl/jax on import, otherwise
    # basicConfig is a no-op and our INFO logs get dropped (root stays at WARNING).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )
    logger.setLevel(logging.INFO)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config_name", nargs="?", help="Registered TrainConfig name (e.g. pi05_libero_rep).")
    p.add_argument("--checkpoint", help="Path to a params checkpoint dir (…/<step>/params).")
    p.add_argument("--batch-size", type=int, default=None, help="Override config.batch_size for feature extraction.")
    p.add_argument("--train-batches", type=int, default=40)
    p.add_argument("--val-batches", type=int, default=40)
    p.add_argument("--rep-dim", type=int, default=128, help="Probe output dim.")
    p.add_argument("--probe-steps", type=int, default=3000)
    p.add_argument("--probe-batch", type=int, default=256, help="Probe minibatch (= #InfoNCE negatives).")
    p.add_argument("--out", default="rep_layer_probe.png", help="Output plot path (None to skip).")
    p.add_argument("--smoke-test", action="store_true", help="Run the probe-fit logic on synthetic data and exit.")
    args = p.parse_args()

    if args.smoke_test:
        run_smoke_test(args)
        return
    if not args.config_name or not args.checkpoint:
        p.error("config_name and --checkpoint are required (unless --smoke-test)")
    if args.out == "None":
        args.out = None
    run_probe(args)


if __name__ == "__main__":
    main()
