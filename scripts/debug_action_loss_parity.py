"""Minimal parity check: plain pi05 vs frozen pi05rep action loss.

Loads the SAME pretrained checkpoint (pi05_libero) into both a plain `Pi0Config`
model and a `Pi0RepConfig` model, then pushes the *same* batches through
`compute_loss` on each and prints the per-batch action loss side by side.

No optimizer, no training state, no freeze filter (freezing is irrelevant to a
pure forward pass). The rep model's action loss reads only the current-obs half,
so we pass `future_observation = observation`; that half is discarded and cannot
affect the number we care about.

Run:
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/debug_action_loss_parity.py

Expectation: if the redesign left the action path equivalent to plain pi05, the
two columns match. A ~10x gap localizes the bug to the pi05rep forward.
"""

import dataclasses

import flax.nnx as nnx
import flax.traverse_util as tu
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.weight_loaders as _weight_loaders

CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero/params"
BATCH_SIZE = 8
NUM_BATCHES = 20
SEED = 0


def load_pretrained(model, params_path):
    """Merge the checkpoint's weights into `model` in place (params present in the
    checkpoint are overwritten; params absent from it keep their fresh init)."""
    graphdef, state = nnx.split(model)
    pure = state.to_pure_dict()
    # CheckpointWeightLoader.load returns exactly the checkpoint params that exist
    # in the model tree (plus any missing lora), i.e. the subset to overwrite with.
    partial = _weight_loaders.CheckpointWeightLoader(params_path).load(pure)
    partial = tu.unflatten_dict(
        {k: v for k, v in tu.flatten_dict(partial).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )
    state.replace_by_pure_dict(partial)
    return nnx.merge(graphdef, state)


def kernel_norm(model):
    return float(jnp.linalg.norm(model.action_out_proj.kernel.value.astype(jnp.float32)))


def main():
    rng = jax.random.key(SEED)

    plain_cfg = _config.get_config("pi05_libero")
    rep_cfg = _config.get_config("pi05_crl_libero_full_finetune_frozen")

    print("Building + loading plain pi05 ...")
    plain = plain_cfg.model.create(jax.random.key(1))
    plain = load_pretrained(plain, CHECKPOINT)
    plain.eval()

    print("Building + loading pi05rep (all weights from same checkpoint) ...")
    rep = rep_cfg.model.create(jax.random.key(2))
    rep = load_pretrained(rep, CHECKPOINT)
    rep.eval()

    # Sanity: the frozen action projection must be identical in both models.
    print(f"action_out_proj ||W||  plain={kernel_norm(plain):.6f}  rep={kernel_norm(rep):.6f}")

    # Data loader from the plain config -> yields (Observation, Actions) tuples.
    # Small batch, no shuffle, single worker for determinism.
    small_cfg = dataclasses.replace(plain_cfg, batch_size=BATCH_SIZE, num_workers=0)
    loader = _data_loader.create_data_loader(
        small_cfg, shuffle=False, num_batches=NUM_BATCHES, skip_norm_stats=False
    )

    print(f"\n{'batch':>5}  {'plain pi05':>12}  {'pi05rep':>12}  {'ratio':>8}")
    plain_all, rep_all = [], []
    for i, (obs, actions) in enumerate(loader):
        rng_b = jax.random.fold_in(rng, i)

        plain_loss = jnp.mean(plain.compute_loss(rng_b, obs, actions, train=False))

        # rep: future_observation = obs (its half is discarded); episode_id irrelevant to action loss.
        _, info = rep.compute_loss(rng_b, obs, obs, actions, train=False, episode_id=None)
        rep_loss = jnp.mean(info["action_loss"])

        p, r = float(plain_loss), float(rep_loss)
        plain_all.append(p)
        rep_all.append(r)
        print(f"{i:>5}  {p:>12.6f}  {r:>12.6f}  {r / p:>8.3f}")

    print(
        f"\nmean over {NUM_BATCHES} batches:  "
        f"plain={np.mean(plain_all):.6f}  rep={np.mean(rep_all):.6f}  "
        f"ratio={np.mean(rep_all) / np.mean(plain_all):.3f}"
    )
    print(
        "\nInterpretation:\n"
        "  ratio ~= 1.0  -> rep action path is equivalent to plain pi05 (bug is elsewhere / not a regression).\n"
        "  ratio ~= 10   -> the pi05rep forward is degrading the action loss; inspect embed/positions/masks there.\n"
        "  Note: noise/time are sampled independently (rep samples over 2B), so expect small\n"
        "  per-batch jitter; only a large systematic gap is meaningful."
    )


if __name__ == "__main__":
    main()
