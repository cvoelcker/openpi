import os

os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0SFConfig


def _make(batch: int = 2):
    cfg = Pi0SFConfig(
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m",
    )
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=batch)
    actions = jnp.zeros((batch, cfg.action_horizon, cfg.action_dim), dtype=jnp.float32)
    return cfg, model, obs, actions


def _sf_batch(obs, actions, **extra):
    return {
        "observation": obs,
        "future_observation": obs,
        "actions": actions,
        "next_observation": obs,
        "next_actions": actions,
        **extra,
    }


def test_compute_loss_shapes_and_finite():
    _, model, obs, actions = _make()
    loss, info = model.compute_loss(jax.random.key(1), _sf_batch(obs, actions), train=True)
    assert jnp.isfinite(jnp.mean(loss))
    # rep_loss aliases sf_loss so the train loop's shared logging works unchanged.
    for k in [
        "action_loss",
        "sf_loss",
        "rep_loss",
        "phi_norm",
        "psi_norm",
        "sf_td_resid",
        "phi_mix_entropy",
        "psi_mix_entropy",
    ]:
        assert k in info and bool(jnp.isfinite(info[k]))
    np.testing.assert_allclose(np.array(info["rep_loss"]), np.array(info["sf_loss"]))


def test_psi_is_unit_norm():
    # psi(s) is the prefix/state feature, L2-normalized (anti-collapse). Under CRL's convention
    # the unit-norm token is psi, not phi.
    _, model, obs, actions = _make()
    _, info = model.compute_loss(jax.random.key(1), _sf_batch(obs, actions), train=True)
    np.testing.assert_allclose(np.array(info["psi_norm"]), 1.0, atol=1e-3)


def test_loss_is_differentiable():
    # Backward must flow through the SF head (and action head) with finite, nonzero gradient.
    _, model, obs, actions = _make()

    def loss_only(m):
        loss, _ = m.compute_loss(jax.random.key(1), _sf_batch(obs, actions), train=True)
        return jnp.mean(loss)

    grads = nnx.grad(loss_only)(model)
    arrs = [x for x in jax.tree.leaves(grads) if isinstance(x, jax.Array)]
    assert arrs
    assert all(bool(jnp.isfinite(x).all()) for x in arrs)
    total = sum(float(jnp.sum(jnp.abs(x))) for x in arrs)
    assert total > 0.0


def test_ortho_metrics_in_info():
    # The orthonormality (anti-collapse) loss on psi and the off-diagonal cosine collapse metric
    # must be logged. psi_offdiag_cossim is a mean absolute cosine over distinct-state pairs, so
    # it lies in [0, 1] (1.0 == collapsed to a single direction).
    _, model, obs, actions = _make(batch=4)
    _, info = model.compute_loss(jax.random.key(1), _sf_batch(obs, actions), train=True)
    for k in ["ortho_loss", "psi_offdiag_cossim"]:
        assert k in info and bool(jnp.isfinite(info[k]))
    c = float(info["psi_offdiag_cossim"])
    assert 0.0 <= c <= 1.0 + 1e-4


def test_compute_loss_jit_traceable():
    # The real train loop jits train_step, so compute_loss must trace cleanly under jax.jit.
    # The ortho off-diagonal reduction must use masking, not boolean array indexing
    # (gram[~eye]) — the latter has a value-dependent shape and raises
    # NonConcreteBooleanIndexError under jit. Eager tests do NOT catch this; this one does.
    _, model, obs, actions = _make(batch=4)
    graphdef, state = nnx.split(model)
    batch = _sf_batch(obs, actions)

    @jax.jit
    def run(state):
        m = nnx.merge(graphdef, state)
        loss, info = m.compute_loss(jax.random.key(1), batch, train=True)
        return jnp.mean(loss), info["ortho_loss"], info["psi_offdiag_cossim"]

    loss, ortho, cos = run(state)
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(ortho))
    assert 0.0 <= float(cos) <= 1.0 + 1e-4


def test_psi_trained_only_by_ortho():
    # Decoupling / anti-collapse: psi is stop-gradded in the TD target, so its ONLY gradient
    # source is the orthonormality loss. With sf_ortho_coeff=0 the psi head params get zero
    # gradient; with sf_ortho_coeff>0 they get nonzero gradient. This proves psi and phi no
    # longer share the TD objective (which is what made the constant-psi fixed point reachable).
    from jax.tree_util import keystr, tree_flatten_with_path

    psi_parts = ("psi_head", "psi_mix", "psi_proj")

    def psi_grad_mass(ortho_coeff):
        cfg = Pi0SFConfig(
            action_dim=32, action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m",
            sf_ortho_coeff=ortho_coeff,
        )
        model = cfg.create(jax.random.key(0))
        obs = cfg.fake_obs(batch_size=4)
        actions = jnp.zeros((4, cfg.action_horizon, cfg.action_dim), dtype=jnp.float32)

        def loss_only(m):
            loss, _ = m.compute_loss(jax.random.key(1), _sf_batch(obs, actions), train=True)
            return jnp.mean(loss)

        grads = nnx.grad(loss_only)(model)
        total = 0.0
        for path, val in tree_flatten_with_path(grads)[0]:
            if isinstance(val, jax.Array) and any(p in keystr(path) for p in psi_parts):
                total += float(jnp.sum(jnp.abs(val)))
        return total

    assert psi_grad_mass(0.0) == pytest.approx(0.0, abs=1e-6)
    assert psi_grad_mass(1.0) > 1e-6


def test_next_is_pad_masks_bootstrap():
    # With identical current/next inputs, phi(s,a) == phi(s',a') == V. Masking the bootstrap
    # (next_is_pad=True) changes the TD target from psi + gamma*V to psi alone, so sf_loss must differ.
    _, model, obs, actions = _make()
    b = obs.state.shape[0]

    def sf_loss(next_is_pad):
        _, info = model.compute_loss(jax.random.key(1), _sf_batch(obs, actions, next_is_pad=next_is_pad), train=True)
        return float(info["sf_loss"])

    loss_bootstrap = sf_loss(jnp.zeros((b,), dtype=bool))
    loss_terminal = sf_loss(jnp.ones((b,), dtype=bool))
    assert abs(loss_bootstrap - loss_terminal) > 1e-6


def test_compute_loss_logs_z_goal_frac():
    # With future_observation provided, compute_loss samples z (mix of B(future) + random) and
    # must report the realized goal fraction. train_goal_ratio defaults to 0.5.
    _, model, obs, actions = _make()
    _, info = model.compute_loss(jax.random.key(1), _sf_batch(obs, actions), train=True)
    assert "z_goal_frac" in info
    assert 0.0 <= float(info["z_goal_frac"]) <= 1.0


def test_get_state_representations_shape_and_norm():
    cfg, model, obs, _ = _make()
    b = obs.state.shape[0]
    psi, phi, kv_cache, prefix_mask, prefix_len = model.get_state_representations(obs)
    assert psi.shape == (b, cfg.rep_dim)
    # Default phi_input="state_action": phi is action-dependent and not available here.
    assert phi is None
    np.testing.assert_allclose(np.array(jnp.linalg.norm(psi, axis=-1)), 1.0, atol=1e-3)


def test_get_phi_representation_shapes():
    cfg, model, obs, actions = _make()
    b = obs.state.shape[0]
    _, _, kv_cache, prefix_mask, prefix_len = model.get_state_representations(obs)
    obs_pp = _model.preprocess_observation(None, obs, train=False)
    timestep = jnp.zeros((b,), dtype=jnp.float32)
    phi, action_hidden, v_t = model.get_phi_representation(obs_pp, actions, timestep, kv_cache, prefix_mask, prefix_len)
    assert phi.shape == (b, cfg.rep_dim)
    assert v_t.shape == (b, cfg.action_horizon, cfg.action_dim)


def test_adarms_cond_depends_on_z():
    # z conditions F through the action-expert AdaRMS pathway: embed_suffix folds z_proj(z) into
    # adarms_cond, while B(s)=psi (prefix) stays z-free. We assert at the adarms_cond level rather
    # than end-to-end on phi because gemma's AdaRMS is zero-initialized (AdaLN-Zero), so the whole
    # conditioning pathway — time included — is a no-op on a freshly-built model until trained.
    cfg, model, obs, actions = _make()
    b = obs.state.shape[0]
    obs_pp = _model.preprocess_observation(None, obs, train=False)
    t = jnp.zeros((b,), dtype=jnp.float32)
    unit = jnp.ones((b, cfg.rep_dim), dtype=jnp.float32) / (cfg.rep_dim**0.5)
    *_, cond_pos = model.embed_suffix(obs_pp, actions, t, z=unit)
    *_, cond_neg = model.embed_suffix(obs_pp, actions, t, z=-unit)
    *_, cond_none = model.embed_suffix(obs_pp, actions, t)
    # z changes adarms_cond, and z=None preserves the original (time-only) conditioning.
    assert not np.allclose(np.array(cond_pos), np.array(cond_neg), atol=1e-4)
    assert not np.allclose(np.array(cond_pos), np.array(cond_none), atol=1e-4)


def test_compute_velocity_step_shape():
    cfg, model, obs, actions = _make()
    b = obs.state.shape[0]
    kv_cache, _prefix_hidden, prefix_mask = model.get_prefix_cache(obs)
    obs_pp = _model.preprocess_observation(None, obs, train=False)
    v_t = model.compute_velocity_step(obs_pp, actions, 1.0, kv_cache, prefix_mask)
    assert v_t.shape == (b, cfg.action_horizon, cfg.action_dim)


def _count_params(model):
    return sum(int(x.size) for x in jax.tree.leaves(nnx.state(model)) if isinstance(x, jax.Array))


@pytest.mark.manual  # slow: builds two 2B-LoRA backbones on CPU
def test_paligemma_lora_rank_is_honored():
    # If the rank knob is wired into __init__, two models with different paligemma_lora_rank have
    # different total parameter counts. If the knob is ignored, both fall back to the variant
    # default (16) and counts match.
    def build(rank):
        cfg = Pi0SFConfig(
            action_dim=32,
            action_horizon=16,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m",
            paligemma_lora_rank=rank,
            paligemma_lora_alpha=float(rank),
        )
        return _count_params(cfg.create(jax.random.key(0)))

    assert build(4) != build(16)


def test_frozen_backbone_freeze_filter():
    # With action_loss_coeff=0 and rep_backbone_grad_scale=0 the backbone gets no gradient;
    # the freeze filter must keep only the phi/psi head params (heads, mixes, projections).
    cfg = Pi0SFConfig(
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
    assert len(trainable) > 0
    rep_parts = ("phi_head", "psi_head", "phi_mix", "psi_mix", "phi_proj", "psi_proj")
    assert all(any(any(t in str(p) for t in rep_parts) for p in path) for path in trainable)


def _make_ema(batch: int = 2, decay: float = 0.5):
    cfg = Pi0SFConfig(
        action_dim=32, action_horizon=16,
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m",
        sf_target_ema=decay,
    )
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=batch)
    actions = jnp.zeros((batch, cfg.action_horizon, cfg.action_dim), dtype=jnp.float32)
    return cfg, model, obs, actions


def test_target_ema_disabled_by_default():
    # Default (sf_target_ema=None) builds no target leaves and leaves compute_loss on the
    # online-bootstrap path, so all existing behavior is unchanged.
    from openpi.models.pi05sf import _SFTarget

    _, model, _, _ = _make()
    assert model.sf_target_ema is None
    assert len(jax.tree.leaves(nnx.state(model, _SFTarget))) == 0


def test_target_params_are_nonparam_float32():
    # Target weights must be a distinct non-Param Variable (so the optimizer's
    # trainable_filter = All(Param, ...) never touches them) and float32 (so the frozen-param
    # bf16 cast, which is Param-path based, leaves EMA precision intact).
    from openpi.models.pi05sf import _SFTarget

    _, model, _, _ = _make_ema()
    tgt = jax.tree.leaves(nnx.state(model, _SFTarget))
    assert len(tgt) > 0
    assert all(v.dtype == jnp.float32 for v in tgt)
    # none of the target leaves are Params
    param_paths = set(nnx.state(model, nnx.Param).flat_state().keys())
    target_paths = set(nnx.state(model, _SFTarget).flat_state().keys())
    assert param_paths.isdisjoint(target_paths)


def test_target_state_flattens_with_string_sep():
    # Weight-loading at init flattens the pure state dict with a string separator
    # (flax.traverse_util.flatten_dict(state.to_pure_dict(), sep="/") in train.py). Every
    # state-path component must therefore be a str. A list target container makes nnx emit
    # integer path components (list indices), which sep.join() rejects -> init crash. The
    # string-keyed dict container (see _relkey) guards that regression.
    import flax.traverse_util as traverse_util

    _, model, _, _ = _make_ema()
    pure = nnx.state(model).to_pure_dict()
    flat = traverse_util.flatten_dict(pure, sep="/")  # raises TypeError on any int key
    assert any("_phi_head_target" in k for k in flat)  # target leaves are actually present


def test_sync_target_networks_hardcopies_online():
    # sync_target_networks() (called once at init after the checkpoint merge) hard-copies the
    # online phi head into the phi target, so a warm-started phi head is reflected in the target.
    # Perturb every online phi_proj param, sync, and each target leaf must now equal its online
    # counterpart. This is the warm-start correctness the old path-pairing test guarded.
    from openpi.models.pi05sf import _SFTarget

    _, model, _, _ = _make_ema()
    assert len(jax.tree.leaves(nnx.state(model, _SFTarget))) > 0
    model.phi_proj.kernel.value = model.phi_proj.kernel.value + 3.0  # online now != target

    model.sync_target_networks()

    proj_flat = nnx.state(model.phi_proj).flat_state()
    for rp in model._phi_proj_relpaths:
        np.testing.assert_allclose(
            np.array(model._phi_proj_target[model._relkey(rp)].value),
            np.array(proj_flat[rp].value),
            atol=1e-5,
        )


def test_update_target_networks_moves_target_toward_online():
    # update_target_networks() (called each train_step after the optimizer step) EMAs the phi
    # target toward the online phi head IN PLACE. Perturb an online param so target != online,
    # step once, and the target must move toward (not reach) online, while the nnx.state -> merge
    # roundtrip preserves the _SFTarget type. This is the exact path that is easy to get wrong.
    from openpi.models.pi05sf import _SFTarget

    _, model, _, _ = _make_ema(decay=0.5)
    graphdef = nnx.graphdef(model)
    n_targets = len(jax.tree.leaves(nnx.state(model, _SFTarget)))  # all target leaves (mix+head+proj)
    model.phi_proj.kernel.value = model.phi_proj.kernel.value + 2.0  # online now differs from target
    proj_flat = nnx.state(model.phi_proj).flat_state()
    before = {k: jnp.asarray(t.value) for k, t in model._phi_proj_target.items()}

    model.update_target_networks()  # in-place EMA, exactly as train_step invokes it

    # roundtrip preserves every _SFTarget leaf (count unchanged through state -> merge)
    model2 = nnx.merge(graphdef, nnx.state(model))
    assert len(jax.tree.leaves(nnx.state(model2, _SFTarget))) == n_targets

    moved = 0
    for rp in model._phi_proj_relpaths:
        k = model._relkey(rp)
        online = proj_flat[rp].value
        b, a = before[k], model._phi_proj_target[k].value
        if not bool(jnp.allclose(b, a)):
            moved += 1
            assert float(jnp.abs(a - online).mean()) < float(jnp.abs(b - online).mean())  # closer to online
    assert moved >= 1


def test_compute_loss_with_target_ema_finite():
    # Integration: with the target bootstrap enabled, compute_loss runs end-to-end and stays
    # finite (phi_next comes from the merged target head rather than the online head).
    _, model, obs, actions = _make_ema()
    loss, info = model.compute_loss(jax.random.key(1), _sf_batch(obs, actions), train=True)
    assert jnp.isfinite(jnp.mean(loss))
    assert bool(jnp.isfinite(info["sf_loss"]))


def test_compute_loss_raises_without_next_inputs():
    _, model, obs, actions = _make()
    with pytest.raises(ValueError, match="next_observation"):
        model.compute_loss(
            jax.random.key(1),
            {"observation": obs, "future_observation": obs, "actions": actions},
            train=True,
        )
