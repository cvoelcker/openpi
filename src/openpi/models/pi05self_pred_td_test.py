import os

os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.pi0_config import Pi0SPConfig
from openpi.models.pi0_config import Pi0SPTDConfig


def _make(batch: int = 4, **overrides):
    cfg = Pi0SPTDConfig(
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


def _td_batch(obs, actions, *, success, done, **extra):
    """`done` drives next_is_pad: the anchor's action chunk reaches the end of the episode, so
    the row takes the label instead of bootstrapping. is_terminal rides along for diagnostics."""
    batch = actions.shape[0]
    return {
        "observation": obs,
        "actions": actions,
        "next_observation": obs,
        "episode_success": jnp.full((batch,), success, dtype=jnp.bool_),
        "next_is_pad": jnp.full((batch,), done, dtype=jnp.bool_),
        "is_terminal": jnp.full((batch,), done, dtype=jnp.bool_),
        **extra,
    }


def test_compute_loss_shapes_and_finite():
    cfg, model, obs, actions = _make()
    batch = _td_batch(obs, actions, success=True, done=False)
    loss, info = model.compute_loss(jax.random.key(1), batch, train=True)
    assert loss.shape == (actions.shape[0], cfg.action_horizon)
    assert jnp.isfinite(jnp.mean(loss))
    for k in ["sp_loss", "sigreg_loss", "value_loss", "value_mean", "value_terminal_frac", "rep_loss"]:
        assert k in info
        assert bool(jnp.isfinite(info[k]))


def test_value_is_a_probability():
    _, model, obs, actions = _make()
    _, info = model.compute_loss(jax.random.key(1), _td_batch(obs, actions, success=True, done=False), train=False)
    assert 0.0 <= float(info["value_mean"]) <= 1.0


def _value_loss(*, success: bool, gamma: float, done: bool, **overrides) -> float:
    _, model, obs, actions = _make(value_gamma=gamma, **overrides)
    batch = _td_batch(obs, actions, success=success, done=done)
    _, info = model.compute_loss(jax.random.key(1), batch, train=False)
    return float(info["value_loss"])


def test_terminal_rows_use_the_reward_and_not_the_bootstrap():
    # On terminal rows the target is the label, so it must move with `episode_success` and be
    # completely insensitive to the discount. Failure terminals are only kept in the loss with
    # value_truncated_failure off -- the default drops them as timeouts.
    won = _value_loss(success=True, gamma=0.99, done=True)
    lost = _value_loss(success=False, gamma=0.99, done=True, value_truncated_failure=False)
    assert won != pytest.approx(lost, abs=1e-4)
    np.testing.assert_allclose(won, _value_loss(success=True, gamma=0.5, done=True), rtol=1e-5)


def test_truncated_failure_drops_the_terminal_row():
    # A failed episode ended on a timeout: its last frame carries no reward and no successor,
    # so it must leave the loss entirely rather than be regressed to 0.
    assert _value_loss(success=False, gamma=0.99, done=True) == pytest.approx(0.0, abs=1e-9)
    assert _value_loss(success=False, gamma=0.99, done=True, value_truncated_failure=False) > 1e-3
    # Successful terminals and every non-terminal row are untouched by the flag.
    for success, done in [(True, True), (True, False), (False, False)]:
        kept = _value_loss(success=success, gamma=0.99, done=done)
        assert kept == pytest.approx(
            _value_loss(success=success, gamma=0.99, done=done, value_truncated_failure=False),
            rel=1e-6,
        )


def test_non_terminal_rows_bootstrap_and_ignore_the_label():
    # Off-terminal the reward is zero: the target is gamma * V(s'), so the discount matters and
    # the episode label does not.
    won = _value_loss(success=True, gamma=0.99, done=False)
    np.testing.assert_allclose(won, _value_loss(success=False, gamma=0.99, done=False), rtol=1e-6)
    assert won != pytest.approx(_value_loss(success=True, gamma=0.1, done=False), abs=1e-4)


def test_value_target_is_bounded_by_the_label_and_the_discount():
    # A terminal failure pins the target at 0; every other target is gamma * V(s') <= gamma.
    _, model, obs, actions = _make(value_gamma=0.5, value_truncated_failure=False)
    _, info = model.compute_loss(jax.random.key(1), _td_batch(obs, actions, success=False, done=True), train=False)
    assert float(info["value_terminal_frac"]) == 1.0
    assert 0.0 <= float(info["value_at_terminal"]) <= 1.0


def _phi_head_grads(**overrides) -> dict[str, np.ndarray]:
    _, model, obs, actions = _make(**overrides)
    batch = _td_batch(obs, actions, success=True, done=True)

    def loss_only(m):
        loss, _ = m.compute_loss(jax.random.key(1), batch, train=False)
        return jnp.mean(loss)

    leaves, _ = jax.tree_util.tree_flatten_with_path(nnx.grad(loss_only)(model))
    return {
        jax.tree_util.keystr(path): np.array(leaf)
        for path, leaf in leaves
        if isinstance(leaf, jax.Array) and "phi_head" in jax.tree_util.keystr(path)
    }


def test_stop_grad_phi_leaves_the_rep_gradient_untouched():
    # With value_stop_grad_phi the head is a pure probe: the phi head's gradient must match the
    # one it would get with the value loss switched off entirely.
    probe = _phi_head_grads(value_stop_grad_phi=True)
    off = _phi_head_grads(value_loss_coeff=0.0)
    assert probe
    assert set(probe) == set(off)
    for key, value in probe.items():
        np.testing.assert_allclose(value, off[key], rtol=1e-5, atol=1e-6)


def test_shaping_phi_changes_the_rep_gradient():
    # The default (value_stop_grad_phi=False) must actually push the TD signal into phi,
    # otherwise the whole point of the head is lost.
    def total(grads):
        return sum(float(np.abs(g).sum()) for g in grads.values())

    assert total(_phi_head_grads()) != pytest.approx(total(_phi_head_grads(value_loss_coeff=0.0)), rel=1e-4)


def test_done_is_driven_by_next_is_pad_not_is_terminal():
    # A chunk that reaches the end of the episode is treated as terminal even when the anchor
    # is not the last frame: it takes the label, so the loss must be insensitive to the
    # discount. Bootstrapping it would spend a full gamma on a sub-chunk gap.
    def loss(*, gamma, is_terminal):
        _, model, obs, actions = _make(value_gamma=gamma)
        batch = actions.shape[0]
        td = _td_batch(obs, actions, success=True, done=True)
        td["is_terminal"] = jnp.full((batch,), is_terminal, dtype=jnp.bool_)
        _, info = model.compute_loss(jax.random.key(1), td, train=False)
        return float(info["value_loss"])

    np.testing.assert_allclose(loss(gamma=0.99, is_terminal=False), loss(gamma=0.1, is_terminal=False), rtol=1e-5)
    # And is_terminal itself must not enter the loss at all.
    np.testing.assert_allclose(loss(gamma=0.99, is_terminal=False), loss(gamma=0.99, is_terminal=True), rtol=1e-6)


def test_value_loss_depends_on_next_is_pad():
    # done rows take the label, non-done rows bootstrap -- the two must not coincide.
    lo = _value_loss(success=True, gamma=0.99, done=False)
    hi = _value_loss(success=True, gamma=0.99, done=True)
    assert lo != pytest.approx(hi, rel=1e-4)


def test_normalize_input_changes_the_value_head_output():
    # phi is unnormalized in SP, so gating the L2 step must actually change what the head sees.
    values = []
    for normalize in (True, False):
        _, model, obs, actions = _make(value_normalize_input=normalize)
        batch = _td_batch(obs, actions, success=True, done=False)
        _, info = model.compute_loss(jax.random.key(1), batch, train=False)
        values.append(float(info["value_mean"]))
    assert values[0] != pytest.approx(values[1], abs=1e-5)


def test_sarsa_requires_next_actions():
    _, model, obs, actions = _make(value_action_conditioned=True)
    batch = _td_batch(obs, actions, success=True, done=False)
    with pytest.raises(ValueError, match="next_actions"):
        model.compute_loss(jax.random.key(1), batch, train=False)


def test_sarsa_bootstrap_depends_on_next_actions():
    # The whole point of SARSA over TD: the target is Q(s', a') for the dataset's own a', so
    # changing a' must move the loss. Non-terminal rows, otherwise the bootstrap is masked.
    _, model, obs, actions = _make(value_action_conditioned=True)
    losses = []
    for fill in (0.0, 1.0):
        batch = _td_batch(obs, actions, success=True, done=False, next_actions=jnp.full_like(actions, fill))
        _, info = model.compute_loss(jax.random.key(1), batch, train=False)
        losses.append(float(info["value_loss"]))
    assert losses[0] != pytest.approx(losses[1], rel=1e-5)


def test_action_gap_is_reported_only_in_sarsa_mode():
    _, model, obs, actions = _make(value_action_conditioned=True)
    batch = _td_batch(obs, actions, success=True, done=False, next_actions=actions)
    _, info = model.compute_loss(jax.random.key(1), batch, train=False)
    assert "value_action_gap" in info
    assert bool(jnp.isfinite(info["value_action_gap"]))

    _, model_v, obs, actions = _make()
    _, info_v = model_v.compute_loss(jax.random.key(1), _td_batch(obs, actions, success=True, done=False), train=False)
    assert "value_action_gap" not in info_v


def test_q_head_never_touches_the_sp_forward_model():
    # The action conditioning must stay inside the value head. forward_proj is the rep
    # objective's latent forward model; if TD reached it, the value loss would reshape what
    # Pi0SP predicts with. Holds at the default coeff, not just under value_stop_grad_phi.
    def forward_proj_grads(**overrides):
        _, model, obs, actions = _make(value_action_conditioned=True, **overrides)
        batch = _td_batch(obs, actions, success=True, done=True, next_actions=actions)

        def loss_only(m):
            loss, _ = m.compute_loss(jax.random.key(1), batch, train=False)
            return jnp.mean(loss)

        leaves, _ = jax.tree_util.tree_flatten_with_path(nnx.grad(loss_only)(model))
        return {
            jax.tree_util.keystr(path): np.array(leaf)
            for path, leaf in leaves
            if isinstance(leaf, jax.Array) and "forward_proj" in jax.tree_util.keystr(path)
        }

    on = forward_proj_grads()
    off = forward_proj_grads(value_loss_coeff=0.0)
    assert on
    assert set(on) == set(off)
    for key, value in on.items():
        np.testing.assert_allclose(value, off[key], rtol=1e-5, atol=1e-6)


def test_compute_loss_raises_without_success_labels():
    _, model, obs, actions = _make()
    batch = {"observation": obs, "actions": actions, "next_observation": obs}
    with pytest.raises(ValueError, match="episode_success"):
        model.compute_loss(jax.random.key(1), batch, train=False)


def test_frozen_backbone_keeps_value_head_trainable():
    cfg = Pi0SPTDConfig(
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
    assert any("value_head" in p for p in paths), paths
    allowed = (
        "phi_head",
        "psi_head",
        "phi_mix",
        "psi_mix",
        "phi_proj",
        "psi_proj",
        "forward_proj",
        "sigreg_proj",
        "value_head",
    )
    assert all(any(a in p for a in allowed) for p in paths), paths


def test_value_head_params_are_absent_from_a_plain_sp_model():
    # The head lives only on the subclass, so existing SP checkpoints stay loadable as Pi0SP.
    sp = Pi0SPConfig(action_dim=32, action_horizon=16, paligemma_variant="gemma_2b_lora")
    abstract = nnx.eval_shape(sp.create, jax.random.key(0))
    paths = ["/".join(str(p) for p in path) for path, _ in nnx.state(abstract, nnx.Param).flat_state()]
    assert not any("value_head" in p for p in paths)
