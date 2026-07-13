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


def test_compute_loss_shapes_and_finite():
    _, model, obs, actions = _make()
    loss, info = model.compute_loss(
        jax.random.key(1), obs, obs, actions,
        next_observation=obs, next_actions=actions, train=True,
    )
    assert jnp.isfinite(jnp.mean(loss))
    for k in ["action_loss", "sf_loss", "phi_norm", "psi_norm", "sf_td_resid"]:
        assert k in info and bool(jnp.isfinite(info[k]))


def test_psi_is_unit_norm():
    # psi(s) is the prefix/state feature, L2-normalized (anti-collapse). Under CRL's convention
    # the unit-norm token is psi, not phi.
    _, model, obs, actions = _make()
    _, info = model.compute_loss(
        jax.random.key(1), obs, obs, actions,
        next_observation=obs, next_actions=actions, train=True,
    )
    np.testing.assert_allclose(np.array(info["psi_norm"]), 1.0, atol=1e-3)


def test_loss_is_differentiable():
    # Backward must flow through the SF head (and action head) with finite, nonzero gradient.
    _, model, obs, actions = _make()

    def loss_only(m):
        loss, _ = m.compute_loss(
            jax.random.key(1), obs, obs, actions,
            next_observation=obs, next_actions=actions, train=True,
        )
        return jnp.mean(loss)

    grads = nnx.grad(loss_only)(model)
    arrs = [x for x in jax.tree.leaves(grads) if isinstance(x, jax.Array)]
    assert arrs
    assert all(bool(jnp.isfinite(x).all()) for x in arrs)
    total = sum(float(jnp.sum(jnp.abs(x))) for x in arrs)
    assert total > 0.0


def test_next_is_pad_masks_bootstrap():
    # With identical current/next inputs, phi(s,a) == phi(s',a') == V. Masking the bootstrap
    # (next_is_pad=True) changes the TD target from psi + gamma*V to psi alone, so sf_loss must differ.
    _, model, obs, actions = _make()
    b = obs.state.shape[0]

    def sf_loss(next_is_pad):
        _, info = model.compute_loss(
            jax.random.key(1), obs, obs, actions,
            next_observation=obs, next_actions=actions, next_is_pad=next_is_pad, train=True,
        )
        return float(info["sf_loss"])

    loss_bootstrap = sf_loss(jnp.zeros((b,), dtype=bool))
    loss_terminal = sf_loss(jnp.ones((b,), dtype=bool))
    assert abs(loss_bootstrap - loss_terminal) > 1e-6


def test_compute_loss_logs_z_goal_frac():
    # With future_observation provided, compute_loss samples z (mix of B(future) + random) and
    # must report the realized goal fraction. train_goal_ratio defaults to 0.5.
    _, model, obs, actions = _make()
    _, info = model.compute_loss(
        jax.random.key(1), obs, obs, actions,
        next_observation=obs, next_actions=actions, train=True,
    )
    assert "z_goal_frac" in info
    assert 0.0 <= float(info["z_goal_frac"]) <= 1.0


def test_get_psi_representation_shape_and_norm():
    cfg, model, obs, _ = _make()
    b = obs.state.shape[0]
    psi, kv_cache, prefix_mask, prefix_len = model.get_psi_representation(obs)
    assert psi.shape == (b, cfg.sf_dim)
    np.testing.assert_allclose(np.array(jnp.linalg.norm(psi, axis=-1)), 1.0, atol=1e-3)


def test_get_phi_representation_shapes():
    cfg, model, obs, actions = _make()
    b = obs.state.shape[0]
    _, kv_cache, prefix_mask, prefix_len = model.get_psi_representation(obs)
    obs_pp = _model.preprocess_observation(None, obs, train=False)
    timestep = jnp.zeros((b,), dtype=jnp.float32)
    phi, action_hidden, v_t = model.get_phi_representation(
        obs_pp, actions, timestep, kv_cache, prefix_mask, prefix_len
    )
    assert phi.shape == (b, cfg.sf_dim)
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
    unit = jnp.ones((b, cfg.sf_dim), dtype=jnp.float32) / (cfg.sf_dim ** 0.5)
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
            action_dim=32, action_horizon=16,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m",
            paligemma_lora_rank=rank, paligemma_lora_alpha=float(rank),
        )
        return _count_params(cfg.create(jax.random.key(0)))

    assert build(4) != build(16)


def test_compute_loss_raises_without_next_inputs():
    _, model, obs, actions = _make()
    with pytest.raises(ValueError, match="next_observation"):
        model.compute_loss(jax.random.key(1), obs, obs, actions, train=True)
