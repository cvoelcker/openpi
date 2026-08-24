"""Tests for the RLT-encoder + FB-objective model.

The load-bearing one is `test_grad_a_q_is_nonzero`. Under a frozen backbone
(`rep_backbone_grad_scale=0.0`) the stop_gradient sits on the ONLY path from `a` to F, so
forgetting `grad_scale_override=1.0` inside `grad_a_q` makes it return exactly zeros -- guided
sampling then silently degenerates to unguided at every guidance scale, with no error, no NaN,
and a perfectly plausible-looking eval. That bug has already cost this study one full sweep.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi05fb_rlt as _rlt

# Dummy gemma variants (width 64, depth 4) keep the LLM tiny. Note the SigLIP tower is NOT
# variant-scaled -- it is always So400m/14 -- so the prefix is still 3*256 + max_token_len
# tokens. That dominates test runtime; keep batch sizes small.
DUMMY = {
    "paligemma_variant": "dummy",
    "action_expert_variant": "dummy",
    "action_horizon": 4,
    "rlt_enc_width": 32,
    "rlt_depth": 2,
    "rlt_mlp_dim": 64,
    "rlt_num_heads": 4,
    "rep_dim": 32,
}


def _config(**extra):
    return _rlt.Pi0FBRLTConfig(**{**DUMMY, **extra})


def _trainable_filter(config):
    """Mirror of `TrainConfig.trainable_filter`, which lives on the TRAIN config, not the model
    config. The `nnx.Param` conjunct is what keeps the non-Param `_FBTarget` EMA leaves out of
    the optimizer even though the freeze regex matches their paths."""
    import flax.nnx as nnx

    return nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter()))


def _rand_obs(config, batch, key):
    """Randomized observation. NOT `config.fake_obs()`, which is all-ones: identical rows make
    every across-batch diagnostic (feat_std, eff_rank, pairwise_cos) degenerate, and a previous
    commit in this repo had to fix exactly that class of fixture bug.

    THE PROMPT TOKENS MUST VARY ACROSS THE BATCH, and randomizing the images is not enough.
    `siglip._Module` defaults to `head_zeroinit=True`, so under RANDOM init (no weight_loader,
    which is the case for every dummy-variant test) the head's Dense kernel is all zeros and
    EVERY image token is the same constant vector no matter what the pixels are. The image path
    only carries information once real PaliGemma weights are loaded. The token embedder, by
    contrast, is `nn.initializers.normal()`, so varying token ids does produce varying prefix
    embeddings at random init -- which is what makes `B(s)` non-constant here.
    """
    keys = jax.random.split(key, 5)
    names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    images = {
        n: jax.random.uniform(keys[i], (batch, 224, 224, 3), minval=-1.0, maxval=1.0) for i, n in enumerate(names)
    }
    return _model.Observation(
        images=images,
        image_masks={n: jnp.ones((batch,), dtype=bool) for n in names},
        state=jax.random.normal(keys[3], (batch, config.action_dim)),
        tokenized_prompt=jax.random.randint(keys[4], (batch, config.max_token_len), 1, 1000, dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((batch, config.max_token_len), dtype=bool),
    )


def _rand_batch(config, batch, key):
    k = jax.random.split(key, 4)
    return {
        "observation": _rand_obs(config, batch, k[0]),
        "next_observation": _rand_obs(config, batch, k[1]),
        "actions": jax.random.normal(k[2], (batch, config.action_horizon, config.action_dim)),
        "next_actions": jax.random.normal(k[3], (batch, config.action_horizon, config.action_dim)),
        "next_is_pad": jnp.zeros((batch,), dtype=bool),
    }


# ---------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------


def test_effective_rank_bounds():
    """Participation ratio: ~d for isotropic data, ~1 for a rank-1 collapse."""
    key = jax.random.key(0)
    d = 16
    iso = jax.random.normal(key, (512, d))
    assert _rlt._effective_rank(iso) > 0.7 * d  # noqa: SLF001

    direction = jax.random.normal(jax.random.key(1), (d,))
    collapsed = jax.random.normal(jax.random.key(2), (512, 1)) * direction[None, :]
    assert _rlt._effective_rank(collapsed) < 1.5  # noqa: SLF001


def test_effective_rank_is_batch_capped():
    """With b < d the centered Gram has rank <= b-1, so eff_rank cannot exceed it. This is the
    whole reason B_ortho_err is unreadable when rep_dim > batch_size."""
    x = jax.random.normal(jax.random.key(0), (8, 512))
    assert _rlt._effective_rank(x) <= 8.0  # noqa: SLF001


def test_rep_health_flags_collapse():
    """A batch-constant representation must read as fully collapsed on every axis. eff_rank is
    exactly 0 there (the centered Gram is identically zero), which is the signature that caught
    a dead-image-path fixture bug -- see _rand_obs."""
    ones = jnp.ones((8, 16))
    info = _rlt._rep_health(ones, "x")  # noqa: SLF001
    assert float(info["rep/x/feat_std"]) == pytest.approx(0.0, abs=1e-6)
    assert float(info["rep/x/dead_frac"]) == pytest.approx(1.0)
    assert float(info["rep/x/pairwise_cos_abs"]) == pytest.approx(1.0, abs=1e-4)
    assert float(info["rep/x/eff_rank"]) == pytest.approx(0.0, abs=1e-6)
    assert float(info["rep/x/nonfinite_frac"]) == 0.0


# ---------------------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------------------


def test_rep_dim_may_not_exceed_encoder_width():
    """rank(E[BB^T]) <= enc_width + 1 at ANY batch size, so d_fb > enc_width makes the
    orthonormality target structurally unreachable and wastes the extra dimensions."""
    with pytest.raises(ValueError, match="exceeds an encoder width"):
        _config(rlt_enc_width=32, rep_dim=64)


def test_b_input_must_be_state():
    with pytest.raises(ValueError, match="action-independent"):
        _config(B_input="state_action")


def test_width_must_divide_heads():
    with pytest.raises(ValueError, match="divisible"):
        _config(rlt_enc_width=32, rlt_num_heads=5, rep_dim=32)


def test_requires_goal_data():
    """Routes training through rl_data_loader, which supplies next_observation/next_actions."""
    assert _config().requires_goal_data is True


# ---------------------------------------------------------------------------------------
# The head
# ---------------------------------------------------------------------------------------


def test_head_shape_and_mask_invariance():
    """Masked tokens must not influence the RL token. If they do, the 256 zeroed right-wrist
    image tokens and every language pad slot leak into B(s)."""
    head = _rlt._RLTHead(  # noqa: SLF001
        enc_width=32, depth=2, mlp_dim=64, num_heads=4, max_tokens=16, embed_dtype="float32"
    )
    b, t, d_in = 3, 10, 8
    mem = jax.random.normal(jax.random.key(0), (b, t, d_in))
    mask = jnp.concatenate([jnp.ones((b, 6), bool), jnp.zeros((b, 4), bool)], axis=1)
    variables = head.init(jax.random.key(1), mem, mask)

    z, stats = head.apply(variables, mem, mask)
    assert z.shape == (b, 32)
    assert jnp.all(jnp.isfinite(z))

    # Perturb ONLY the masked positions; the output must not move.
    mem2 = mem.at[:, 6:].add(100.0)
    z2, _ = head.apply(variables, mem2, mask)
    np.testing.assert_allclose(np.asarray(z), np.asarray(z2), rtol=1e-5, atol=1e-5)

    # Per-block diagnostics are present and finite.
    for i in range(2):
        for k in ("rl_attn_entropy", "rl_attn_self", "resid_norm"):
            assert jnp.isfinite(stats[f"b{i}/{k}"]), f"b{i}/{k}"
    # Attention entropy of the RL row is bounded by log(T+1) over the valid tokens.
    assert 0.0 <= float(stats["b0/rl_attn_entropy"]) <= float(jnp.log(t + 1.0)) + 1e-4


# ---------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return _config().create(jax.random.key(0))


def test_freeze_filter_selects_exactly_the_heads(model):
    """Pinned rather than trusted to the regex: with rep_backbone_grad_scale=0.0 only the F/B
    encoder and output projections may train. A regex that silently matches nothing would leave
    the whole 3B backbone trainable and OOM at the first step."""
    import flax.nnx as nnx

    config = _config()
    graphdef, state = nnx.split(model)
    del graphdef
    frozen = set(nnx.state(model, config.get_freeze_filter()).flat_state())
    every = set(state.flat_state())
    trainable = every - frozen
    assert trainable, "freeze filter left nothing trainable"
    for path in trainable:
        joined = "/".join(map(str, path))
        assert "F_head" in joined or "B_head" in joined or "F_out" in joined or "B_out" in joined, joined
    # And every head parameter is trainable.
    for path in every:
        joined = "/".join(map(str, path))
        if any(t in joined for t in ("F_head", "B_head", "F_out", "B_out")):
            assert path in trainable, joined


def test_compute_loss_shapes_and_finiteness(model):
    config = _config()
    b = 4
    batch = _rand_batch(config, b, jax.random.key(3))
    loss, info = model.compute_loss(jax.random.key(4), batch, train=True)

    assert loss.shape == (b, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))

    for k, v in info.items():
        assert jnp.all(jnp.isfinite(v)), f"non-finite diagnostic: {k} = {v}"


def test_diagnostics_panel_is_complete(model):
    """If a key disappears the dashboards silently lose a panel, so pin the contract."""
    config = _config()
    _, info = model.compute_loss(jax.random.key(5), _rand_batch(config, 4, jax.random.key(6)), train=True)
    required = {
        "fb_loss",
        "fb_ortho_loss",
        "rep_loss",
        "fb/M_rank_top1",
        "fb/B_sigma2",
        "fb/B_ortho_err",
        "fb/F_B_cos_matched",
        "fb/F_B_cos_shuffled",
        "fb/M_eff_rank",
        "fb/d_over_b",
        "rep/B/eff_rank",
        "rep/B/feat_std",
        "rep/B/dead_frac",
        "rep/F/eff_rank",
        "rep/z_B/eff_rank",
        "rep/z_F/eff_rank",
        "enc/B/b0/rl_attn_entropy",
        "enc/B/b0/rl_attn_self",
        "enc/B/posemb_norm",
        "enc/F/b0/rl_attn_entropy",
        "enc/F/in_proj_norm",
    }
    missing = required - set(info)
    assert not missing, f"missing diagnostics: {sorted(missing)}"


def test_diagnostic_ranges(model):
    config = _config()
    _, info = model.compute_loss(jax.random.key(7), _rand_batch(config, 4, jax.random.key(8)), train=True)
    assert 0.0 <= float(info["rep/B/pairwise_cos_abs"]) <= 1.0
    assert 0.0 <= float(info["rep/B/dead_frac"]) <= 1.0
    assert 0.0 <= float(info["fb/M_rank_top1"]) <= 1.0
    assert 0.0 <= float(info["fb/M_sv_top1_frac"]) <= 1.0 + 1e-5
    assert float(info["rep/B/eff_rank"]) > 0.0
    assert float(info["enc/B/b0/rl_attn_self"]) >= 0.0
    # d_fb / batch, the number that says how degenerate the bilinear term is.
    assert float(info["fb/d_over_b"]) == pytest.approx(config.rep_dim / 4.0)


def test_terminal_masking_changes_the_loss(model):
    """next_is_pad must actually gate the bootstrap."""
    config = _config()
    batch = _rand_batch(config, 4, jax.random.key(9))
    l0, _ = model.compute_loss(jax.random.key(10), batch, train=False)
    l1, _ = model.compute_loss(jax.random.key(10), {**batch, "next_is_pad": jnp.ones((4,), bool)}, train=False)
    assert not jnp.allclose(l0, l1)


def test_ema_targets_are_excluded_from_the_trainable_set(model):
    """The freeze regex `.*(F|B)_(head|out).*` ALSO matches the EMA target containers
    (`_B_head_t`, `_F_out_t`, ...), so they are not in freeze_filter. That is deliberate --
    being in freeze_filter would cast them to bfloat16, which loses the (1-decay) EMA increments
    entirely -- and they are kept out of the OPTIMIZER by `trainable_filter`'s `nnx.Param`
    conjunct instead, since `_FBTarget` is not a Param. Pin both halves of that contract."""
    import flax.nnx as nnx

    config = _config()
    trainable = set(nnx.state(model, _trainable_filter(config)).flat_state())
    frozen = set(nnx.state(model, config.get_freeze_filter()).flat_state())
    target_paths = [p for p in nnx.state(model).flat_state() if any(str(c).startswith(("_B_", "_F_")) for c in p)]
    assert target_paths, "no EMA target leaves found -- did _build_targets run?"
    for p in target_paths:
        assert p not in trainable, f"EMA target is trainable: {p}"
        assert p not in frozen, f"EMA target is in freeze_filter (would be cast to bf16): {p}"


def test_gradients_reach_only_the_heads(model):
    """A gradient must exist for the encoders and be exactly absent for the backbone.

    Goes through the SAME split -> merge -> nnx.value_and_grad(DiffState(0, trainable_filter))
    dance as `train.py::train_step`. Calling `nnx.grad` on the live model instead raises
    "Arrays leaves are not supported" -- under `nnx_bridge.ToNNX` some linen leaves are plain
    arrays rather than Variables until they have been round-tripped through nnx.split, and the
    filter then selects one. train_step never hits it because it always differentiates a merged
    state; this mirrors that.
    """
    import flax.nnx as nnx

    config = _config()
    batch = _rand_batch(config, 4, jax.random.key(11))
    trainable = _trainable_filter(config)

    # Functional split: differentiate a plain State pytree with jax.grad rather than handing the
    # live module to nnx.grad(DiffState(...)). The latter raises "Arrays leaves are not
    # supported" here -- under nnx_bridge.ToNNX some linen leaves are raw arrays rather than
    # Variables, and DiffState's filter selects one. train_step never trips this because it
    # differentiates a State it split itself, which is exactly what this mirrors.
    graphdef, params, rest = nnx.split(model, trainable, ...)

    def loss_fn(p):
        m = nnx.merge(graphdef, p, rest)
        loss, _ = m.compute_loss(jax.random.key(12), batch, train=True)
        return jnp.mean(loss)

    grads = jax.grad(loss_fn)(params)
    flat = grads.flat_state()
    assert flat, "no gradients produced"
    for path in flat:
        joined = "/".join(map(str, path))
        assert any(t in joined for t in ("F_head", "B_head", "F_out", "B_out")), joined
    total = sum(float(jnp.sum(jnp.abs(v.value))) for v in flat.values())
    assert total > 0.0, "encoder gradients are identically zero"


# ---------------------------------------------------------------------------------------
# Deployment -- the one that matters
# ---------------------------------------------------------------------------------------


def test_grad_a_q_is_nonzero(model):
    """THE regression test. See the module docstring.

    `_rep` applies rep_backbone_grad_scale (0.0 => stop_gradient) to the ONLY path from `a` to F.
    `grad_a_q` must pass grad_scale_override=1.0 or this returns exact zeros and every guided
    eval silently becomes an unguided eval.
    """
    config = _config()
    b = 2
    obs = _rand_obs(config, b, jax.random.key(13))
    actions = jax.random.normal(jax.random.key(14), (b, config.action_horizon, config.action_dim))

    _, _, kv_cache, prefix_mask, prefix_len = model.get_state_representations(obs)
    pre = _model.preprocess_observation(None, obs, train=False)
    z_r = jax.random.normal(jax.random.key(15), (config.rep_dim,))

    grad, q = model.grad_a_q(pre, actions, kv_cache, prefix_mask, prefix_len, z_r)
    assert grad.shape == actions.shape
    assert q.shape == (b,)
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.max(jnp.abs(grad))) > 0.0, "grad_a_q is identically zero -- guidance is inert"


def test_test_time_adapt_builds_z_r(model):
    """The mixin gates on psi_input == 'state' before building z_r; Pi0FBRLT overwrites
    psi_input in __init__ so B is treated as psi."""
    from openpi.models import rep_grad_guidance as _rgg

    config = _config()
    k = 5
    samples = _rgg.RewardSamples(
        observation=_rand_obs(config, k, jax.random.key(16)),
        reward=jnp.asarray([0.0, 0.0, 1.0, 0.0, 1.0]),
    )
    z_r = model.test_time_adapt(samples)
    assert z_r.shape == (config.rep_dim,)
    assert jnp.all(jnp.isfinite(z_r))
    # normalize_z_r defaults True for this config.
    assert float(jnp.linalg.norm(z_r)) == pytest.approx(1.0, abs=1e-4)


def test_target_networks_track_and_are_float32(model):
    """EMA targets must be a non-Param Variable type so the optimizer never updates them and the
    frozen-param bf16 cast leaves them float32 -- bf16 loses the (1-decay) increments entirely."""
    assert model.fb_target_ema is not None
    leaf = next(iter(model._B_out_t.values()))  # noqa: SLF001
    assert leaf.value.dtype == jnp.float32
    before = jnp.asarray(leaf.value)
    model.sync_target_networks()  # hard copy; should be a no-op right after init
    np.testing.assert_allclose(np.asarray(before), np.asarray(leaf.value), rtol=1e-6)


# ---------------------------------------------------------------------------------
# EMA target updates.
#
# `scripts/train.py` drives these through a duck-typed `getattr(model, "update_target_networks")`
# hook, and nothing fails loudly if `_blend` skips a group or misapplies the decay -- the target
# just stops lagging and `fb_target_ema` becomes decorative. These pin the arithmetic, the
# coverage of all four groups, and the float32 invariant behind `_FBTarget`.
#
# Each builds its OWN model: the `model` fixture is module-scoped and these mutate weights.
# ---------------------------------------------------------------------------------


def _fresh(**extra):
    return _config(**extra).create(jax.random.key(0))


def _bump(module, delta):
    """Add `delta` to every parameter of an nnx module, through the public state API so the same
    helper works for `nnx.Linear` (B_out/F_out) and the ToNNX-bridged encoder (B_head/F_head)."""
    import flax.nnx as nnx

    nnx.update(module, jax.tree.map(lambda x: x + delta, nnx.state(module)))


def _group_leaves(model, group):
    return getattr(model, f"_{group}_t")


_GROUPS = ("B_head", "B_out", "F_head", "F_out")


def test_update_target_networks_moves_target_toward_online():
    """decay=0.5 must move each target leaf exactly halfway to the online value. A target that
    jumps all the way (decay applied to the wrong operand) is not a target network at all."""
    model = _fresh(fb_target_ema=0.5)
    model.sync_target_networks()
    online_before = {k: jnp.asarray(v.value) for k, v in _group_leaves(model, "B_out").items()}

    _bump(model.B_out, 1.0)
    model.update_target_networks()

    for k, target in _group_leaves(model, "B_out").items():
        expected = 0.5 * online_before[k] + 0.5 * (online_before[k] + 1.0)
        np.testing.assert_allclose(np.asarray(target.value), np.asarray(expected), rtol=1e-6)


def test_update_target_networks_moves_every_target_group():
    """`_blend` loops over ("F","B") x ("head","out"). Dropping one pair leaves that half of the
    bilinear bootstrap pinned at initialization forever, which no other assertion would catch."""
    model = _fresh(fb_target_ema=0.5)
    model.sync_target_networks()
    before = {g: {k: jnp.asarray(v.value) for k, v in _group_leaves(model, g).items()} for g in _GROUPS}

    for g in _GROUPS:
        _bump(getattr(model, g), 1.0)
    model.update_target_networks()

    for g in _GROUPS:
        moved = [not jnp.allclose(before[g][k], v.value) for k, v in _group_leaves(model, g).items() if v.value.size]
        assert moved, f"target group {g} has no leaves to check"
        assert all(moved), f"target group {g} did not move"


def test_sync_target_networks_hard_copies_after_drift():
    """`sync_target_networks` is `_blend(0.0)` and runs once after the checkpoint merge, when the
    online weights are real and the freshly-built targets are not. Testing it only at init (where
    it is a no-op) would not distinguish a hard copy from doing nothing."""
    model = _fresh(fb_target_ema=0.99)
    _bump(model.F_out, 3.0)
    model.sync_target_networks()

    import flax.nnx as nnx

    online = nnx.state(model.F_out).flat_state()
    for rp in model._F_out_rp:  # noqa: SLF001
        target = _group_leaves(model, "F_out")[model._relkey(rp)]  # noqa: SLF001
        np.testing.assert_allclose(np.asarray(target.value), np.asarray(online[rp].value), rtol=1e-6)


def test_update_target_networks_keeps_targets_float32():
    """The whole reason `_FBTarget` is a distinct non-Param Variable: the frozen-param bf16 cast
    matches Param paths, and at bf16 a decay of 0.99 rounds the (1-decay) increment away to
    nothing, so the target would silently never move."""
    model = _fresh(fb_target_ema=0.99)
    model.sync_target_networks()

    import flax.nnx as nnx

    nnx.update(model.B_out, jax.tree.map(lambda x: (x + 1.0).astype(jnp.bfloat16), nnx.state(model.B_out)))
    model.update_target_networks()

    for target in _group_leaves(model, "B_out").values():
        assert target.value.dtype == jnp.float32


def test_update_target_networks_is_a_noop_without_ema():
    """`fb_target_ema=None` skips target construction entirely, so the train loop's unconditional
    hook must still be safe to call."""
    model = _fresh(fb_target_ema=None)
    assert not hasattr(model, "_B_out_t")
    model.update_target_networks()  # must not raise
    model.sync_target_networks()


def test_train_loop_target_hook_is_discoverable():
    """scripts/train.py finds this by `getattr`, not by an interface or registration, so a rename
    would disable the EMA with no import error and no test failure anywhere else."""
    model = _fresh(fb_target_ema=0.99)
    hook = getattr(model, "update_target_networks", None)
    assert callable(hook)
    assert callable(getattr(model, "sync_target_networks", None))
