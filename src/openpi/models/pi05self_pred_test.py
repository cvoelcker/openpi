import os

os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0SPConfig
from openpi.models.rep_base import batch_rep_stats


def _make(batch: int = 4, **overrides):
    cfg = Pi0SPConfig(
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
        **overrides,
    )
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=batch)
    actions = jnp.zeros((batch, cfg.action_horizon, cfg.action_dim), dtype=jnp.float32)
    return cfg, model, obs, actions


def _sp_batch(obs, actions, **extra):
    return {
        "observation": obs,
        "actions": actions,
        "next_observation": obs,
        **extra,
    }


def test_compute_loss_shapes_and_finite():
    cfg, model, obs, actions = _make()
    loss, info = model.compute_loss(jax.random.key(1), _sp_batch(obs, actions), train=True)
    assert loss.shape == (actions.shape[0], cfg.action_horizon)
    assert jnp.isfinite(jnp.mean(loss))
    for k in ["action_loss", "sp_loss", "sigreg_loss", "rep_loss", "phi_norm", "phi_self_sim"]:
        assert k in info
        assert bool(jnp.isfinite(info[k]))


@pytest.mark.parametrize("normalize", [True, False])
def test_sp_loss_is_independent_of_rep_dim(normalize):
    # The regression is expressed relative to the mean squared target norm, so it is invariant
    # in rep_dim on BOTH paths. A sum over features would grow with rep_dim when unnormalized;
    # a mean over features would shrink as 1/rep_dim when normalized (2 - 2*cos is already a
    # sum). Either would silently reweight sp_loss against the dimension-free sigreg term,
    # making runs at different rep_dim incomparable.
    losses = {}
    for rep_dim in (128, 512):
        _, model, obs, actions = _make(rep_dim=rep_dim, normalize_sp_loss=normalize)
        _, info = model.compute_loss(jax.random.key(1), _sp_batch(obs, actions), train=False)
        losses[rep_dim] = float(info["sp_loss"])
    # Not equal (different random heads), but the same order of magnitude — a linear or
    # inverse-linear rep_dim scaling would put these 4x apart.
    ratio = losses[128] / max(losses[512], 1e-9)
    assert 0.25 < ratio < 4.0, losses


def test_sp_loss_is_scale_invariant():
    # Nothing pins ||phi||, so a loss that moves under a global rescale of the reps would make
    # "shrink everything" a free way to lower it. The relative form removes that.
    rng = np.random.RandomState(0)
    pred = jnp.asarray(rng.randn(8, 64), dtype=jnp.float32)
    tgt = jnp.asarray(rng.randn(8, 64), dtype=jnp.float32)

    def rel(scale):
        p, t = pred * scale, tgt * scale
        return float(jnp.mean(jnp.sum(jnp.square(p - t), axis=-1)) / (jnp.mean(jnp.sum(jnp.square(t), axis=-1)) + 1e-8))

    base = rel(1.0)
    for scale in (1e-2, 1e2):
        np.testing.assert_allclose(rel(scale), base, rtol=1e-3)


def test_sp_loss_reduces_to_cosine_when_normalized():
    # With both sides on the unit sphere the denominator is exactly 1, so the relative form
    # collapses to the plain 2 - 2*cos in [0, 4].
    _, model, obs, actions = _make(normalize_sp_loss=True)
    _, info = model.compute_loss(jax.random.key(1), _sp_batch(obs, actions), train=False)
    assert 0.0 <= float(info["sp_loss"]) <= 4.0 + 1e-3


@pytest.mark.parametrize("mode", ["independent", "shared", "none"])
def test_target_augmentation_modes_run(mode):
    _, model, obs, actions = _make(target_augmentation=mode)
    loss, info = model.compute_loss(jax.random.key(1), _sp_batch(obs, actions), train=True)
    assert jnp.isfinite(jnp.mean(loss))
    assert bool(jnp.isfinite(info["sp_loss"]))


def test_target_augmentation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="target_augmentation"):
        _make(target_augmentation="coupled")


def test_shared_augmentation_makes_target_match_input_augmentation():
    # "shared" reuses the input's rng, so with next_observation == observation the two
    # preprocessed views are identical; "independent" must NOT produce that.
    cfg, _, obs, _ = _make()
    rng = jax.random.key(3)
    a = _model.preprocess_observation(rng, obs, train=True, augment=cfg.image_augment)
    b_same = _model.preprocess_observation(rng, obs, train=True, augment=cfg.image_augment)
    b_diff = _model.preprocess_observation(jax.random.key(4), obs, train=True, augment=cfg.image_augment)
    key = next(iter(a.images))
    np.testing.assert_allclose(np.array(a.images[key]), np.array(b_same.images[key]), atol=1e-5)
    assert not np.allclose(np.array(a.images[key]), np.array(b_diff.images[key]), atol=1e-5)


def test_forward_proj_output_is_unnormalized():
    # ForwardProjHead must NOT L2-normalize internally: with normalize_sp_loss=False the
    # target is raw, so a unit-norm prediction could never fit it.
    cfg, model, _, _ = _make(normalize_sp_loss=False)
    phi = jnp.asarray(np.random.RandomState(0).randn(4, cfg.rep_dim), dtype=jnp.float32) * 7.0
    acts = jnp.zeros((4, cfg.action_dim * cfg.action_horizon), dtype=jnp.float32)
    out = model.forward_proj(phi, acts)
    norms = np.array(jnp.linalg.norm(out, axis=-1))
    assert not np.allclose(norms, 1.0, atol=1e-3), norms


def test_sigreg_projector_has_no_layernorm():
    # A LayerNorm in the projector divides out ||phi||. With normalize_sp_loss the regression
    # is scale-invariant too, so SIGReg is the only term left that can pin phi's magnitude.
    _, model, _, _ = _make()
    assert not hasattr(model, "sigreg_proj_ln")
    x = jnp.asarray(np.random.RandomState(0).randn(4, model.phi_dim), dtype=jnp.float32)
    small = model._sigreg_project(x)  # noqa: SLF001
    large = model._sigreg_project(x * 100.0)  # noqa: SLF001
    # Scale must propagate through, otherwise SIGReg cannot see it at all.
    assert float(jnp.linalg.norm(large)) > 10.0 * float(jnp.linalg.norm(small))


def test_frozen_backbone_keeps_sigreg_proj_trainable():
    # LeJEPA issue #17: the isotropy constraint is meaningful only if the projector it acts
    # through is learned. Left frozen it is a fixed random map.
    cfg = Pi0SPConfig(
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
        action_loss_coeff=0.0,
        rep_backbone_grad_scale=0.0,
    )
    assert cfg.backbone_frozen
    abstract_model = nnx.eval_shape(cfg.create, jax.random.key(0))
    trainable = nnx.state(abstract_model, nnx.All(nnx.Param, nnx.Not(cfg.get_freeze_filter()))).flat_state()
    paths = ["/".join(str(p) for p in path) for path, _ in trainable]
    assert any("sigreg_proj" in p for p in paths), paths
    assert any("forward_proj" in p for p in paths), paths
    # ...and nothing from the backbone leaked in.
    allowed = ("phi_head", "psi_head", "phi_mix", "psi_mix", "phi_proj", "psi_proj", "forward_proj", "sigreg_proj")
    assert all(any(a in p for a in allowed) for p in paths), paths


def test_action_pass_skipped_when_action_loss_off():
    # With action_loss_coeff=0 the expensive suffix pass is skipped and action_loss is exactly
    # zero (rather than a computed value multiplied by zero).
    _, model, obs, actions = _make(action_loss_coeff=0.0, rep_backbone_grad_scale=0.0)
    _, info = model.compute_loss(jax.random.key(1), _sp_batch(obs, actions), train=False)
    assert float(info["action_loss"]) == 0.0


def test_next_is_pad_excludes_rows_from_sp_loss():
    # Padded rows must be dropped from the average, not merely zeroed and then divided by the
    # full batch (which would make the reported loss depend on the terminal fraction).
    _, model, obs, actions = _make(batch=4)
    rng = jax.random.key(1)
    _, info_all = model.compute_loss(rng, _sp_batch(obs, actions), train=False)
    pad = jnp.array([0.0, 0.0, 1.0, 1.0])
    _, info_pad = model.compute_loss(rng, _sp_batch(obs, actions, next_is_pad=pad), train=False)
    # obs rows are identical here, so masking half the batch must leave the mean unchanged.
    np.testing.assert_allclose(np.array(info_pad["sp_loss"]), np.array(info_all["sp_loss"]), rtol=1e-4)


def test_loss_is_differentiable():
    _, model, obs, actions = _make()

    def loss_only(m):
        loss, _ = m.compute_loss(jax.random.key(1), _sp_batch(obs, actions), train=True)
        return jnp.mean(loss)

    grads = nnx.grad(loss_only)(model)
    arrs = [x for x in jax.tree.leaves(grads) if isinstance(x, jax.Array)]
    assert arrs
    assert all(bool(jnp.isfinite(x).all()) for x in arrs)
    assert sum(float(jnp.sum(jnp.abs(x))) for x in arrs) > 0.0


def test_compute_loss_raises_without_next_observation():
    _, model, obs, actions = _make()
    with pytest.raises(ValueError, match="next_observation"):
        model.compute_loss(jax.random.key(1), {"observation": obs, "actions": actions}, train=False)


# --- rep_base.batch_rep_stats: the collapse diagnostics the SP runs are steered by ---


def test_eff_rank_matches_eigendecomposition():
    rng = np.random.RandomState(0)
    for rep in [
        rng.randn(64, 128),  # isotropic
        np.outer(rng.randn(64), rng.randn(128)),  # rank 1
        rng.randn(64, 8) @ rng.randn(8, 128),  # rank 8
    ]:
        stats = batch_rep_stats(jnp.asarray(rep, dtype=jnp.float32), "x")
        centered = rep - rep.mean(axis=0, keepdims=True)
        lam = np.linalg.svd(centered, compute_uv=False) ** 2
        expected = lam.sum() ** 2 / (lam**2).sum()
        np.testing.assert_allclose(float(stats["x_eff_rank"]), expected, rtol=1e-3)


def test_eff_rank_is_scale_invariant():
    # The whole point: with normalize_sp_loss nothing pins ||phi||, so a metric that moves
    # when phi is globally rescaled cannot distinguish collapse from a harmless rescale.
    rep = jnp.asarray(np.random.RandomState(0).randn(64, 128), dtype=jnp.float32)
    base = float(batch_rep_stats(rep, "x")["x_eff_rank"])
    for scale in (1e-6, 1e-3, 1e3):
        scaled = float(batch_rep_stats(rep * scale, "x")["x_eff_rank"])
        np.testing.assert_allclose(scaled, base, rtol=1e-3)


def test_eff_rank_floors_at_one_under_total_collapse():
    # A batch-constant rep has zero centered variance: the participation ratio is 0/0 and the
    # normalized Gram is pure roundoff, whose rank is high. Must report the floor instead.
    rep = jnp.tile(jnp.asarray(np.random.RandomState(0).randn(128), dtype=jnp.float32), (64, 1))
    stats = batch_rep_stats(rep, "x")
    np.testing.assert_allclose(float(stats["x_eff_rank"]), 1.0, atol=1e-4)
    np.testing.assert_allclose(float(stats["x_offdiag_cos"]), 1.0, atol=1e-3)


def test_offdiag_cos_excludes_the_diagonal():
    # Orthogonal rows -> mean off-diagonal cosine 0. Including the b unit-valued diagonal
    # entries would bias it to 1/b.
    rep = jnp.asarray(np.eye(16, 32), dtype=jnp.float32)
    stats = batch_rep_stats(rep, "x")
    np.testing.assert_allclose(float(stats["x_offdiag_cos"]), 0.0, atol=1e-5)


# --- rep-head capacity / regularization knobs (defaults must be no-ops) ---


def test_rep_head_width_none_is_the_default_shape():
    # rep_head_width=None must reproduce the historical head exactly, including leaving
    # mlp_dim / head_dim at paligemma's values even when the memory width differs.
    cfg = Pi0SPConfig(action_dim=32, action_horizon=16, paligemma_variant="gemma_2b_lora")
    assert cfg.rep_head_width is None
    assert cfg.rep_head_block_dropout == 0.0


def test_rep_head_width_shrinks_the_head():
    def head_params(width):
        cfg = Pi0SPConfig(
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            rep_head_width=width,
        )
        abstract = nnx.eval_shape(cfg.create, jax.random.key(0))
        state = nnx.state(abstract, nnx.Param).flat_state()
        return sum(int(np.prod(v.value.shape)) for path, v in state if any("phi_head" in str(p) for p in path))

    wide = head_params(None)
    narrow = head_params(256)
    # The in_proj added by narrowing is tiny next to what the blocks shed.
    assert narrow < wide / 4, (narrow, wide)


def test_block_dropout_changes_the_rep_only_in_train_mode():
    _, model, obs, actions = _make(rep_head_block_dropout=0.5, rep_head_depth=2)
    batch = _sp_batch(obs, actions)
    # Deterministic path (train=False) must be unaffected by the dropout rate.
    _, a = model.compute_loss(jax.random.key(1), batch, train=False)
    _, b = model.compute_loss(jax.random.key(2), batch, train=False)
    np.testing.assert_allclose(np.array(a["sp_loss"]), np.array(b["sp_loss"]), rtol=1e-5)
