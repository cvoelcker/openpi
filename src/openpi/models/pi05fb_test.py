"""Tests for the Forward-Backward model.

The first test is the important one. `fb_loss_terms` is a pure function of arrays precisely so
the OBJECTIVE can be checked against a tabular MDP whose successor measure is known in closed
form -- i.e. so we verify the mathematics, not merely that tensors flow. Everything after it
tests plumbing.

All model tests use the "dummy" Gemma variant (width 64, depth 4) rather than the real
gemma_2b_lora that pi05sf_test.py builds, so this file runs in seconds on CPU instead of minutes.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models.pi0_config import Pi0FBConfig
from openpi.models.pi05fb import fb_loss_terms

# =====================================================================================
# 1. The objective itself, against a tabular MDP
# =====================================================================================


def _tabular_mdp(n_states=6, n_actions=2, gamma=0.9, seed=0):
    """A small ergodic MDP plus the closed-form successor measure of a fixed policy.

    Returns (P, pi, gamma, M, rho, d_sa) where
      P    [S, A, S]   transition kernel
      pi   [S, A]      the policy being evaluated
      M    [S*A, S]    M[(s,a), x] = sum_t gamma^t Pr(s_{t+1} = x | s, a, pi)
      rho  [S]         stationary state distribution under pi (the measure FB integrates against)
      d_sa [S*A]       stationary state-action distribution (what we sample transitions from)
    """
    rng = np.random.default_rng(seed)
    p = rng.random((n_states, n_actions, n_states)) + 0.1
    p /= p.sum(axis=-1, keepdims=True)
    pi = rng.random((n_states, n_actions)) + 0.1
    pi /= pi.sum(axis=-1, keepdims=True)

    sa = n_states * n_actions
    p_sa = p.reshape(sa, n_states)  # Pr(s' | s,a)
    # T[(s,a), (s',a')] = P(s'|s,a) * pi(a'|s')
    t_sa = (p_sa[:, :, None] * pi[None, :, :]).reshape(sa, sa)

    # M = sum_t gamma^t T^t P_sa = (I - gamma T)^{-1} P_sa
    m = np.linalg.solve(np.eye(sa) - gamma * t_sa, p_sa)

    # Stationary distribution of the (s,a) chain -> the sampling distribution; its next-state
    # marginal is the rho the batch's B(s'_j) columns are drawn from.
    evals, evecs = np.linalg.eig(t_sa.T)
    d_sa = np.real(evecs[:, np.argmin(np.abs(evals - 1.0))])
    d_sa = np.abs(d_sa) / np.abs(d_sa).sum()
    rho = d_sa @ p_sa
    return p, pi, gamma, m, rho, d_sa


def _sample_batch(p, pi, d_sa, batch, rng):
    """Draw (s,a) ~ d_sa, s' ~ P(.|s,a), a' ~ pi(.|s'). Returns flat indices."""
    n_states, n_actions, _ = p.shape
    sa_idx = rng.choice(n_states * n_actions, size=batch, p=d_sa)
    s_idx, a_idx = sa_idx // n_actions, sa_idx % n_actions
    probs = p[s_idx, a_idx]
    u = rng.random((batch, 1))
    nxt = (probs.cumsum(axis=-1) < u).sum(axis=-1).clip(0, n_states - 1)
    u2 = rng.random((batch, 1))
    nxt_a = (pi[nxt].cumsum(axis=-1) < u2).sum(axis=-1).clip(0, n_actions - 1)
    return sa_idx, nxt, nxt * n_actions + nxt_a


def _fit_fb(steps=4000, batch=256, rep_dim=6, ortho_coeff=1.0, seed=0):
    p, pi, gamma, m, rho, d_sa = _tabular_mdp(seed=seed)
    n_states, n_actions, _ = p.shape
    sa = n_states * n_actions
    rng = np.random.default_rng(seed + 1)

    key = jax.random.key(seed)
    k1, k2 = jax.random.split(key)
    params = {
        "F": 0.1 * jax.random.normal(k1, (sa, rep_dim)),
        "B": 0.1 * jax.random.normal(k2, (n_states, rep_dim)),
    }

    def loss(params, sa_idx, nxt, nxt_sa):
        f_cur = params["F"][sa_idx]
        b_next = params["B"][nxt]
        f_boot = jax.lax.stop_gradient(params["F"][nxt_sa])
        b_boot = jax.lax.stop_gradient(params["B"][nxt])
        l_fb, l_ortho, _ = fb_loss_terms(
            f_cur,
            b_next,
            f_boot,
            b_boot,
            gamma=gamma,
            not_terminal=jnp.ones((sa_idx.shape[0],)),
        )
        return l_fb + ortho_coeff * l_ortho

    tx = optax.adam(3e-3)
    opt_state = tx.init(params)
    step = jax.jit(lambda pr, os_, a, b, c: _apply(tx, pr, os_, jax.grad(loss)(pr, a, b, c)))

    for _ in range(steps):
        sa_idx, nxt, nxt_sa = _sample_batch(p, pi, d_sa, batch, rng)
        params, opt_state = step(params, opt_state, jnp.asarray(sa_idx), jnp.asarray(nxt), jnp.asarray(nxt_sa))

    # The FB ansatz models the DENSITY of M w.r.t. rho, so the recovery target is M / rho.
    density = np.asarray(m) / np.asarray(rho)[None, :]
    fitted = np.asarray(params["F"] @ params["B"].T)
    rel_err = np.linalg.norm(fitted - density) / np.linalg.norm(density)
    cov = np.asarray(params["B"]).T @ (np.asarray(rho)[:, None] * np.asarray(params["B"]))
    return rel_err, cov, params


def _apply(tx, params, opt_state, grads):
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state


def test_fb_loss_recovers_tabular_successor_measure():
    """THE correctness test: minimizing the implemented loss must reconstruct M / rho.

    If the linear term, the discount, the target, or the sign convention were wrong, the fitted
    bilinear form would converge somewhere else and this fails.
    """
    rel_err, _, _ = _fit_fb()
    assert rel_err < 0.15, f"FB fit did not recover the successor measure: rel_err={rel_err:.3f}"


def test_fb_ortho_drives_unit_variance_not_half():
    """E_rho[B B^T] -> I, i.e. per-dimension variance 1.

    The plug-in (diagonal-included) orthonormality estimator has its argmin at 1/(1 + d/b)
    instead, which is 0.5 whenever rep_dim == batch -- the value sf_psi_ortho_err was pinned at
    in every prior run of this study. This test is what stops that regressing back in.
    """
    _, cov, _ = _fit_fb()
    diag = np.diag(cov)
    assert np.all(diag > 0.7), f"per-dim variance collapsed: {diag}"
    assert abs(diag.mean() - 1.0) < 0.25, f"per-dim variance {diag.mean():.3f}, expected ~1.0 (not ~0.5)"


def test_b_sigma2_is_the_scale_free_orthonormality_metric():
    """B_ortho_err carries a finite-sample floor of ~(d+1)/b and is not comparable across
    settings; B_sigma2 is. Feed an exactly-isotropic unit-variance B and check B_sigma2 reads 1
    while B_ortho_err does not read 0."""
    b, d = 32, 32
    bn = jax.random.normal(jax.random.key(0), (b, d))
    _, _, metrics = fb_loss_terms(bn, bn, bn, bn, gamma=0.9, not_terminal=jnp.ones((b,)))
    assert abs(float(metrics["B_sigma2"]) - 1.0) < 0.3, metrics["B_sigma2"]
    # the floor is real: at d == b it is ~1.0 even for a perfectly isotropic population
    assert float(metrics["B_ortho_err"]) > 0.5, metrics["B_ortho_err"]


def test_fb_loss_linear_term_prevents_collapse():
    """Without the linear term, F = B = 0 minimizes the quadratic term. With it, the loss at the
    zero solution must be beatable -- check the gradient at zero is non-zero."""
    b, d = 8, 4
    zeros = jnp.zeros((b, d))

    def total(f):
        l_fb, _, _ = fb_loss_terms(f, zeros, zeros, zeros, gamma=0.9, not_terminal=jnp.ones((b,)))
        return l_fb

    # dL/dF at F=B=0 is zero because the linear term is bilinear; perturb B instead so the
    # linear term has something to pull on.
    ones = jnp.ones((b, d)) * 0.1

    def total_b(f):
        l_fb, _, _ = fb_loss_terms(f, ones, zeros, zeros, gamma=0.9, not_terminal=jnp.ones((b,)))
        return l_fb

    assert jnp.allclose(jax.grad(total)(zeros), 0.0)
    assert not jnp.allclose(jax.grad(total_b)(zeros), 0.0)


def test_fb_terminal_mask_drops_bootstrap():
    b, d = 6, 3
    rng = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(rng, 4)
    f, bn = jax.random.normal(k1, (b, d)), jax.random.normal(k2, (b, d))
    ft, bt = jax.random.normal(k3, (b, d)), jax.random.normal(k4, (b, d))
    live, _, _ = fb_loss_terms(f, bn, ft, bt, gamma=0.9, not_terminal=jnp.ones((b,)))
    dead, _, _ = fb_loss_terms(f, bn, ft, bt, gamma=0.9, not_terminal=jnp.zeros((b,)))
    # With every transition terminal the bootstrap vanishes, so the residual is M itself.
    no_gamma, _, _ = fb_loss_terms(f, bn, ft, bt, gamma=0.0, not_terminal=jnp.ones((b,)))
    assert not jnp.allclose(live, dead)
    assert jnp.allclose(dead, no_gamma)


def test_rank_metric_is_at_chance_for_random_reps():
    b, d = 64, 8
    k1, k2 = jax.random.split(jax.random.key(0))
    f, bn = jax.random.normal(k1, (b, d)), jax.random.normal(k2, (b, d))
    _, _, metrics = fb_loss_terms(f, bn, f, bn, gamma=0.9, not_terminal=jnp.ones((b,)))
    # Chance is 1/b; allow a generous band, the point is that it is nowhere near 1.
    assert metrics["M_rank_top1"] < 0.2, metrics["M_rank_top1"]


# =====================================================================================
# 2. The model
# =====================================================================================


def _cfg(**kw):
    base = {
        "action_dim": 32,
        "action_horizon": 4,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "rep_dim": 8,
    }
    return Pi0FBConfig(**{**base, **kw})


def _make(batch: int = 4, **kw):
    """NOTE the actions are RANDOM, not zeros.

    A zeros fixture makes F identically zero on the dummy backbone -- the zero action chunk
    propagates through action_in_proj and the RMSNorms to an exactly-zero suffix -- so
    M = F B^T is exactly 0, fb_loss is exactly 0, and every "is it finite" assertion passes
    vacuously. That silently gutted the terminal-masking test, and the identical zero-fixture
    trap already produced one wrong conclusion earlier in this study (an "exactly 0.0 gradient"
    on phi_mix that was the fixture, not the model). test_fixture_is_not_degenerate guards it."""
    cfg = _cfg(**kw)
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=batch)
    actions = 0.5 * jax.random.normal(jax.random.key(7), (batch, cfg.action_horizon, cfg.action_dim))
    return cfg, model, obs, actions


def _fb_batch(obs, actions, **extra):
    return {
        "observation": obs,
        "actions": actions,
        "next_observation": obs,
        "next_actions": actions,
        **extra,
    }


def test_compute_loss_finite_and_shaped():
    cfg, model, obs, actions = _make()
    loss, info = model.compute_loss(jax.random.key(0), _fb_batch(obs, actions), train=True)
    assert loss.shape == (obs.state.shape[0], cfg.action_horizon)
    assert jnp.isfinite(loss).all()
    for k, v in info.items():
        assert jnp.isfinite(jnp.mean(v)), f"{k} is not finite"


def test_fixture_is_not_degenerate():
    """Guard the zero-fixture trap: if F or B is identically zero, M = F B^T vanishes, fb_loss is
    exactly 0, and most assertions in this file become vacuous rather than false."""
    _, model, obs, actions = _make()
    _, info = model.compute_loss(jax.random.key(0), _fb_batch(obs, actions), train=True)
    assert float(info["F_norm"]) > 1e-6, "F is identically zero -- the fixture is degenerate"
    assert float(info["B_norm"]) > 1e-6, "B is identically zero -- the fixture is degenerate"
    assert abs(float(info["fb_loss"])) > 1e-8, "fb_loss is exactly 0 -- nothing is being tested"


def test_log_dict_does_not_shadow_train_loop_metrics():
    """`scripts/train.py` spreads log_dict LAST into its info dict, so these three names would
    silently overwrite the loop's own values."""
    _, model, obs, actions = _make()
    _, info = model.compute_loss(jax.random.key(0), _fb_batch(obs, actions), train=True)
    assert not ({"loss", "grad_norm", "param_norm"} & set(info))


def test_compute_loss_jit_traceable():
    """Guards against value-dependent shapes (boolean indexing) creeping into the loss -- the
    train loop jits it."""
    _, model, obs, actions = _make()
    graphdef, state = nnx.split(model)

    @jax.jit
    def run(state, batch):
        return nnx.merge(graphdef, state).compute_loss(jax.random.key(0), batch, train=True)[0]

    assert jnp.isfinite(run(state, _fb_batch(obs, actions))).all()


def test_terminal_masking_changes_the_loss():
    """Compare fb_loss, NOT the total: at init the total is dominated by l_ortho (which the mask
    does not touch), so the FB term's difference sits under allclose's tolerance and the test
    passes vacuously either way."""
    _, model, obs, actions = _make()
    b = obs.state.shape[0]
    live = model.compute_loss(jax.random.key(0), _fb_batch(obs, actions, next_is_pad=jnp.zeros((b,), bool)))[1]
    dead = model.compute_loss(jax.random.key(0), _fb_batch(obs, actions, next_is_pad=jnp.ones((b,), bool)))[1]
    assert not jnp.allclose(live["fb_loss"], dead["fb_loss"])
    # terminal => no bootstrap => the residual is M itself, so the quadratic term must grow
    assert float(dead["fb_quad"]) > float(live["fb_quad"])


def test_freeze_filter_trains_exactly_the_fb_trios():
    """Pin the trainable set by path. A regex typo that silently widened or narrowed this would
    otherwise only show up as a mysteriously different training run."""
    cfg = _cfg()
    abstract = nnx.eval_shape(cfg.create, jax.random.key(0))
    trainable = nnx.state(abstract, nnx.All(nnx.Param, nnx.Not(cfg.get_freeze_filter()))).flat_state()
    paths = {"/".join(map(str, p)) for p in trainable}
    assert paths, "nothing is trainable"
    for p in paths:
        assert any(tok in p for tok in ("F_head", "F_mix", "F_proj", "B_head", "B_mix", "B_proj")), p
    # every trio member must actually be present
    for tok in ("F_mix", "F_proj", "B_mix", "B_proj"):
        assert any(tok in p for p in paths), f"{tok} is not trainable"
    # and nothing from the backbone
    assert not any("PaliGemma" in p or "llm" in p for p in paths)


def test_gradients_reach_only_fb_params():
    cfg, model, obs, actions = _make()
    graphdef, params, rest = nnx.split(model, nnx.All(nnx.Param, nnx.Not(cfg.get_freeze_filter())), ...)

    def loss_fn(params):
        m = nnx.merge(graphdef, params, rest)
        return jnp.mean(m.compute_loss(jax.random.key(0), _fb_batch(obs, actions), train=True)[0])

    grads = jax.grad(loss_fn)(params)
    total = sum(float(jnp.sum(jnp.abs(g))) for g in jax.tree.leaves(grads))
    assert total > 0.0, "no gradient reached the F/B heads"


def test_grad_a_q_is_nonzero_under_frozen_backbone():
    """The defect that once made guided sampling bit-identical to unguided at every scale: the
    frozen-backbone stop_gradient is the only path from `a` to F, so without
    grad_scale_override=1.0 this returns exactly zeros. Asserting finiteness would not catch it,
    because zero is finite."""
    cfg, model, obs, actions = _make()
    assert cfg.backbone_frozen
    b_rep, _, kv_cache, prefix_mask, prefix_len = model.get_state_representations(obs)
    z_r = jnp.ones((cfg.rep_dim,)) / jnp.sqrt(cfg.rep_dim)
    grad, q = model.grad_a_q(obs, actions, kv_cache, prefix_mask, prefix_len, z_r)
    assert grad.shape == actions.shape
    assert jnp.isfinite(grad).all()
    assert float(jnp.sum(jnp.abs(grad))) > 0.0, "grad_a_q is identically zero"
    assert b_rep.shape == (obs.state.shape[0], cfg.rep_dim)
    assert q.shape == (obs.state.shape[0],)


def test_target_networks_sync_then_lag():
    _, model, obs, actions = _make(fb_target_ema=0.5)
    model.sync_target_networks()
    assert jnp.allclose(model._F_mix_target.value, model.F_mix.value)  # noqa: SLF001
    model.F_mix.value = model.F_mix.value + 1.0
    model.update_target_networks()
    # decay 0.5 -> target moves halfway toward the online value, not all the way.
    expected = 0.5 * (model.F_mix.value - 1.0) + 0.5 * model.F_mix.value
    assert jnp.allclose(model._F_mix_target.value, expected)  # noqa: SLF001


def test_target_ema_none_skips_target_construction():
    _, model, obs, actions = _make(fb_target_ema=None)
    assert not hasattr(model, "_F_mix_target")
    loss, _ = model.compute_loss(jax.random.key(0), _fb_batch(obs, actions), train=True)
    assert jnp.isfinite(loss).all()


def test_include_proprio_changes_the_representation():
    """Proprio must actually reach both heads. With pi05=True and discrete_state_input=False the
    state is otherwise absent from the network entirely."""
    _, model_off, obs, actions = _make(include_proprio=False)
    _, model_on, _, _ = _make(include_proprio=True)
    off = model_off.get_state_representations(obs)[0]
    on = model_on.get_state_representations(obs)[0]
    assert off.shape == on.shape
    # different projection input width -> different params, so just assert both are finite and
    # that the proprio model's B_proj is genuinely wider.
    assert jnp.isfinite(on).all()
    assert model_on.B_proj.kernel.value.shape[0] == model_off.B_proj.kernel.value.shape[0] + 32


def test_include_proprio_loss_runs():
    _, model, obs, actions = _make(include_proprio=True)
    loss, info = model.compute_loss(jax.random.key(0), _fb_batch(obs, actions), train=True)
    assert jnp.isfinite(loss).all()
    assert jnp.isfinite(info["B_ortho_err"])


def test_f_include_prefix_rejects_guidance():
    """Training-only flag: guided sampling reaches F through _suffix_forward, which has no prefix
    hidden states. Better to fail loudly than to silently drop the prefix term at deployment."""
    cfg, model, obs, actions = _make(F_include_prefix=True)
    loss, _ = model.compute_loss(jax.random.key(0), _fb_batch(obs, actions), train=True)
    assert jnp.isfinite(loss).all()
    _, _, kv_cache, prefix_mask, prefix_len = model.get_state_representations(obs)
    with pytest.raises(NotImplementedError):
        model.grad_a_q(obs, actions, kv_cache, prefix_mask, prefix_len, jnp.ones((cfg.rep_dim,)))


def test_b_input_must_be_state():
    with pytest.raises(ValueError, match="action-independent"):
        _cfg(B_input="state_action")


def test_post_init_chains_to_pi0config():
    """Pi0FBConfig.__post_init__ must call super(): Pi0Config's is what fills in max_token_len
    and discrete_state_input from their None sentinels. Shadowing it without super() left both
    at None, which surfaced far away as 'Shapes must be 1D sequences of concrete values of
    integer type, got (1, None)' out of fake_obs() -- and silently broke the real named configs
    too, not just the tests."""
    cfg = _cfg()
    assert cfg.max_token_len == 200, cfg.max_token_len
    assert cfg.discrete_state_input is False
    assert cfg.fake_obs(batch_size=2).tokenized_prompt.shape == (2, 200)


def test_named_fb_configs_are_constructible():
    """The named TrainConfigs must build a model whose freeze_filter literal agrees with the
    model's -- scripts/train.py hard-fails at startup otherwise, after the queue wait.

    fb_config is not registered in config.py yet, so it is looked up directly; this is also the
    only thing exercising that module.
    """
    import openpi.training.misc.fb_config as fb_config

    by_name = {c.name: c for c in fb_config.get_fb_configs()}
    assert set(by_name) == {"pi05_fb_debug_dummy", "pi05_fb_onpolicy", "pi05_fb_onpolicy_proprio"}

    for name, cfg in by_name.items():
        assert cfg.model.max_token_len == 200, name
        assert cfg.model.requires_goal_data, name
        # the same check scripts/train.py::_check_freeze_filter_matches_model performs
        expected = cfg.model.get_freeze_filter()
        abstract = nnx.eval_shape(cfg.model.create, jax.random.key(0))
        got = {"/".join(map(str, p)) for p in nnx.state(abstract, nnx.All(nnx.Param, expected)).flat_state()}
        declared = {
            "/".join(map(str, p)) for p in nnx.state(abstract, nnx.All(nnx.Param, cfg.freeze_filter)).flat_state()
        }
        assert got == declared, name
