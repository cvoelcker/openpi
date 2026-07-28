"""Shared phi/psi representation-head machinery for the pi0 auxiliary-loss variants.

Pi0RepBase extends the Pi0 backbone with dedicated attention-pooling heads (_RepHead) that
read a learned softmax mix over all backbone layers. Both readouts are modular: `phi_input`
and `psi_input` each select their source sequence — "state" (prefix, action-independent) or
"state_action" (suffix, sees the action tokens). Pi0CRL and Pi0SF inherit from this class and
only add their respective loss (InfoNCE / TD successor features).
"""

import dataclasses
import logging
from typing import Literal

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def _feature_dropout(x, rate: float, *, deterministic: bool, rng):
    """Inverted dropout with an explicitly-passed jax key (avoids the flax rng-collection
    machinery, which does not thread cleanly through the nnx bridge). No-op unless training."""
    if rate <= 0.0 or deterministic or rng is None:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)


def _l2_normalize(x, eps: float = 1e-6):
    """Project reps onto the unit sphere (float32). The trained CRL critic is
    exp(logit_scale) * cosine, so every consumer of phi/psi must score normalized reps —
    raw dot products are NOT the trained scoring rule."""
    x = x.astype(jnp.float32)
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def batch_rep_stats(rep, name: str) -> dict[str, at.Array]:
    """Across-batch (axis=0, NOT within-sample) collapse diagnostics for a [b, d] rep.

    `{name}_std` is the mean over features of each feature's std across the batch -> 0 as the
    rep collapses to a batch-constant. `{name}_mean_norm` is the norm of the batch-mean rep; a
    large mean offset can sit under a healthy std, so both are tracked together.

    Those two are scale-DEPENDENT, which makes them ambiguous whenever nothing in the loss
    pins ||rep|| (e.g. Pi0SP with normalize_sp_loss=True, where the regression is scale-
    invariant): a shrinking `_std` can mean either collapse or a harmless global rescale. The
    remaining three are scale-INVARIANT and disambiguate it:

    - `{name}_eff_rank`: participation ratio of the covariance spectrum, (tr S)^2 / tr(S^2),
      in [1, min(b, d)]. An effective count of active directions -> 1 under collapse.
      Computed WITHOUT an eigendecomposition: both traces are Frobenius norms, so this costs
      one b x b Gram matmul (~1e-7 of a training step) instead of an eigensolver, which on
      GPU is a serial kernel and — for the non-symmetric case — has no lowering at all.
    - `{name}_eff_rank_frac`: the same, normalized to [0, 1] so it is comparable across
      rep_dim and batch size.
    - `{name}_offdiag_cos`: mean off-diagonal cosine similarity, -> 1 under directional
      collapse (which a healthy per-feature std can still hide). 0 means the samples are
      mutually orthogonal, which is the healthy end. Read on the RAW rep, so a large shared
      mean offset drives it toward 1 on its own — deep features usually sit in a narrow cone
      off the origin, and nothing here pins the mean. Near 1 is therefore only alarming when
      `_eff_rank` is also near 1; otherwise it is a mean offset (cross-check `_mean_norm`
      against `_std * sqrt(d)`).
    - `{name}_offdiag_cos_centered`: the same after removing the batch mean, i.e. with the
      cone artifact subtracted out. This is the one to read for directional collapse. Its
      healthy value is ~0 (in fact ~-1/(b-1), since centered vectors sum to zero); -> 1 only
      if the samples genuinely vary along a single shared direction.

    `scripts/eval_rep_stats.py` reports the same quantities offline over a pooled sample.
    """
    rep = rep.astype(jnp.float32)
    b = rep.shape[0]

    centered = rep - jnp.mean(rep, axis=0, keepdims=True)
    # (tr S)^2 / tr(S^2) with S the covariance. tr(S) = ||X||_F^2 / b and, since X X^T is
    # symmetric, tr(S^2) = ||X X^T||_F^2 / b^2 — no eigensolver needed. The Gram is taken on
    # the batch axis (b < d here), so this is a small b x b matmul, not a d x d covariance.
    #
    # X is rescaled to unit Frobenius norm first. The b factors and the overall scale cancel
    # algebraically, but NOT numerically: with a fixed epsilon in the denominator, a rep whose
    # magnitude drifts down (which is exactly what happens when nothing pins ||rep||) would
    # have that epsilon take over and report a spurious rank. After the rescale, tr S = 1 and
    # ||X X^T||_F^2 lies in [1 / min(b, d), 1], so the epsilon can never dominate.
    fro_sq = jnp.sum(jnp.square(centered))
    scale_sq = jnp.sum(jnp.square(rep))
    normed = centered / jnp.sqrt(fro_sq + 1e-30)
    gram = normed @ normed.T
    eff_rank = 1.0 / (jnp.sum(jnp.square(gram)) + 1e-30)
    # Total collapse: no across-batch variance relative to the rep's own magnitude. The
    # participation ratio is 0/0 there, and `normed` would be pure roundoff whose rank is
    # high — the exact opposite of the truth. Report the floor of 1 instead.
    eff_rank = jnp.where(fro_sq > 1e-8 * (scale_sq + 1e-30), eff_rank, 1.0)

    def _offdiag_cos(x):
        unit = _l2_normalize(x)
        cos = unit @ unit.T
        # Strip the b unit-valued diagonal entries before averaging.
        return (jnp.sum(cos) - jnp.trace(cos)) / jnp.maximum(b * (b - 1), 1)

    return {
        f"{name}_std": jnp.mean(jnp.std(rep, axis=0)),
        f"{name}_mean_norm": jnp.linalg.norm(jnp.mean(rep, axis=0)),
        f"{name}_eff_rank": eff_rank,
        f"{name}_eff_rank_frac": eff_rank / min(b, rep.shape[1]),
        f"{name}_offdiag_cos": _offdiag_cos(rep),
        f"{name}_offdiag_cos_centered": _offdiag_cos(centered),
    }


class _RepHead(nn.Module):
    """Attention-pooling head: reuses gemma Blocks to pool a layer-mixed backbone
    token sequence into a single vector via a learned query token.

    `config.width` may be narrower than the incoming memory width (see
    Pi0RepBaseConfig.rep_head_width); an input projection then maps the memory down before
    the blocks. `block_dropout` applies feature dropout between blocks, in addition to the
    single `dropout` mask on the pooled output.
    """

    config: _gemma.Config
    embed_dtype: str
    dropout: float = 0.0
    block_dropout: float = 0.0

    @nn.compact
    def __call__(self, memory, memory_mask, *, deterministic: bool = True, dropout_rng=None):
        # memory: (b, t, d); memory_mask: (b, t) bool
        memory = memory.astype(self.embed_dtype)
        if memory.shape[-1] != self.config.width:
            # Narrowed head: project the backbone-width memory down to the head width. Named
            # explicitly so the param path is stable across checkpoints.
            memory = nn.Dense(self.config.width, name="in_proj", dtype=self.embed_dtype)(memory)
        b, t, d = memory.shape
        query = self.param("query", nn.initializers.normal(0.02), (1, 1, d))
        query = jnp.broadcast_to(query, (b, 1, d)).astype(self.embed_dtype)
        seq = jnp.concatenate([memory, query], axis=1)  # (b, t+1, d)
        seq_mask = jnp.concatenate([memory_mask, jnp.ones((b, 1), dtype=jnp.bool_)], axis=1)
        ar_mask = jnp.zeros((t + 1,), dtype=jnp.bool_)  # fully bidirectional among valid tokens
        attn_mask = _pi0.make_attn_mask(seq_mask, ar_mask)[:, None]  # (b, 1, t+1, t+1)
        positions = jnp.cumsum(seq_mask, axis=1) - 1
        for i in range(self.config.depth):
            (seq,), _ = _gemma.Block(configs=(self.config,), name=f"layers_{i}")(
                [seq], None, positions, attn_mask, [None]
            )
            if self.block_dropout > 0.0:
                # fold_in so each block (and the pooled-output mask below) draws its own mask
                # from the single key threaded in by the caller.
                block_rng = None if dropout_rng is None else jax.random.fold_in(dropout_rng, i)
                seq = _feature_dropout(seq, self.block_dropout, deterministic=deterministic, rng=block_rng)
        seq, _ = _gemma.RMSNorm(name="final_norm")(seq, None)
        pooled = seq[:, -1]  # (b, d) — the pooled query row
        return _feature_dropout(pooled, self.dropout, deterministic=deterministic, rng=dropout_rng)


class Pi0RepBase(_pi0.Pi0):
    """Pi0 backbone plus learned phi/psi readout heads over a layer mix.

    Head/projection widths follow each rep's input source: "state" reads the (paligemma-width)
    prefix hidden states, "state_action" the (action-expert-width) suffix hidden states.
    Subclasses set the normalization policy via `_normalize_phi` / `_normalize_psi`.
    """

    # CRL scores cosine similarities, so it normalizes both reps; SF keeps phi unnormalized
    # (successor-feature magnitude matters). Overridden per subclass.
    _normalize_phi: bool = True
    _normalize_psi: bool = True

    def __init__(self, config: pi0_config.Pi0RepBaseConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        paligemma_config = _gemma.get_config(
            config.paligemma_variant,
            lora_rank=config.paligemma_lora_rank,
            lora_alpha=config.paligemma_lora_alpha,
        )
        action_expert_config = _gemma.get_config(
            config.action_expert_variant,
            lora_rank=config.action_expert_lora_rank,
            lora_alpha=config.action_expert_lora_alpha,
        )

        self.action_loss_coeff = config.action_loss_coeff
        # Input each rep is trained on: "state" (prefix) or "state_action" (suffix).
        self.phi_input = config.phi_input
        self.psi_input = config.psi_input
        # Both reps project to rep_dim, so the critic conditioning size is the
        # same regardless of which rep is selected downstream.
        self.phi_dim = config.rep_dim
        self.psi_dim = config.rep_dim
        self.rep_backbone_grad_scale = config.rep_backbone_grad_scale
        self.rep_head_dropout = config.rep_head_dropout
        num_backbone_layers = paligemma_config.depth

        widths = {"state": paligemma_config.width, "state_action": action_expert_config.width}
        psi_mem_width = widths[config.psi_input]
        phi_mem_width = widths[config.phi_input]

        def _head_config(mem_width):
            if config.rep_head_width is None:
                # Historical behavior: head width follows whatever feeds it, while mlp_dim and
                # head_dim stay at paligemma's values regardless. Left exactly as-is — note a
                # "state_action" head already runs at mem_width 1024 with paligemma's 16384
                # mlp_dim, so rescaling here would silently change existing configs.
                return dataclasses.replace(
                    paligemma_config, width=mem_width, depth=config.rep_head_depth, lora_configs={}
                )
            # Narrowed head: scale mlp_dim and head_dim by the same factor so the block keeps
            # gemma's proportions. num_heads is preserved, and _RepHead inserts an in_proj
            # from mem_width down to the head width.
            scale = config.rep_head_width / mem_width
            return dataclasses.replace(
                paligemma_config,
                width=config.rep_head_width,
                depth=config.rep_head_depth,
                mlp_dim=max(int(paligemma_config.mlp_dim * scale), 1),
                head_dim=max(int(paligemma_config.head_dim * scale), 1),
                lora_configs={},
            )

        # The psi trio (psi_head/psi_mix/psi_proj) is optional: subclasses whose loss never
        # reads psi (Pi0SP) set enable_psi_head=False to drop ~1e8 dead params — and their
        # optimizer state in the frozen-backbone regime, where the freeze filter would
        # otherwise mark psi_* trainable. psi construction stays first so the rng draws of
        # the default (enabled) path are unchanged.
        if config.enable_psi_head:
            psi_head = nnx_bridge.ToNNX(
                _RepHead(
                    _head_config(psi_mem_width),
                    config.dtype,
                    dropout=config.rep_head_dropout,
                    block_dropout=config.rep_head_block_dropout,
                )
            )
            psi_head.lazy_init(jnp.zeros((1, 2, psi_mem_width)), jnp.ones((1, 2), dtype=bool), rngs=rngs)
            self.psi_head = psi_head
        else:
            self.psi_head = None
        phi_head = nnx_bridge.ToNNX(
            _RepHead(
                _head_config(phi_mem_width),
                config.dtype,
                dropout=config.rep_head_dropout,
                block_dropout=config.rep_head_block_dropout,
            )
        )
        phi_head.lazy_init(jnp.zeros((1, 2, phi_mem_width)), jnp.ones((1, 2), dtype=bool), rngs=rngs)
        self.phi_head = phi_head

        # Learned softmax mix over all backbone layers, initialized to favor early layers.
        self.psi_mix = nnx.Param(jnp.linspace(1.0, -1.0, num_backbone_layers)) if config.enable_psi_head else None
        self.phi_mix = nnx.Param(jnp.linspace(1.0, -1.0, num_backbone_layers))

        self.phi_proj = nnx.Linear(phi_mem_width, config.rep_dim, rngs=rngs)
        self.psi_proj = nnx.Linear(psi_mem_width, config.rep_dim, rngs=rngs) if config.enable_psi_head else None

        # Lagging-psi target network (BYOL/JEPA-style): when set, psi is never trained by
        # gradient descent — TrainConfig.trainable_filter excludes the psi trio — and is
        # instead EMA-tracked toward phi via update_target_networks() after each optimizer
        # step. Start psi as an exact copy of phi (their fresh inits used different rng draws).
        self.psi_lagging_ema = config.psi_lagging_ema
        if config.psi_lagging_ema is not None:
            self._blend_lagging_psi(0.0)

    def _blend_lagging_psi(self, tau: float):
        """psi <- tau * psi + (1 - tau) * phi across the head/mix/proj trio.

        Valid because psi_lagging_ema enforces psi_input == phi_input, so the two trios are
        architecturally identical. tau=0.0 is a hard sync.
        """
        for online, target in ((self.phi_head, self.psi_head), (self.phi_proj, self.psi_proj)):
            online_state = nnx.state(online, nnx.Param)
            target_state = nnx.state(target, nnx.Param)
            nnx.update(target, jax.tree.map(lambda t, o: tau * t + (1.0 - tau) * o, target_state, online_state))
        self.psi_mix.value = tau * self.psi_mix.value + (1.0 - tau) * self.phi_mix.value

    def sync_target_networks(self):
        """Hard-copy phi into the lagging psi (no-op without psi_lagging_ema).

        The train script calls this once after merging checkpoint weights: a warm-start
        checkpoint may carry a trained phi head (and possibly a psi trained under a different
        objective), and the lagging target must start equal to the online net.
        """
        if self.psi_lagging_ema is not None:
            self._blend_lagging_psi(0.0)

    def update_target_networks(self):
        """One EMA step of the lagging psi (no-op without psi_lagging_ema).

        The train script calls this after every optimizer step, on the updated params.
        """
        if self.psi_lagging_ema is not None:
            self._blend_lagging_psi(self.psi_lagging_ema)

    @staticmethod
    def _scale_grad(x, scale: float):
        # 0 => full stop-gradient (cleanest graph); 1 => identity; else partial.
        if scale == 0.0:
            return jax.lax.stop_gradient(x)
        if scale == 1.0:
            return x
        return scale * x + (1.0 - scale) * jax.lax.stop_gradient(x)

    def _mix_layers(self, hidden, mix_logits, scale: float):
        # hidden: (L, b, t, d); mix_logits: (L,). Returns (b, t, d) float32.
        hidden = self._scale_grad(hidden, scale)
        weights = jax.nn.softmax(mix_logits.astype(jnp.float32))
        return jnp.einsum("l,lbtd->btd", weights, hidden.astype(jnp.float32))

    def _compute_rep(
        self,
        which: Literal["phi", "psi"],
        prefix_hidden,
        suffix_hidden,
        prefix_mask,
        suffix_mask,
        *,
        deterministic: bool = True,
        dropout_rng=None,
    ):
        """Compute the raw (unnormalized) phi or psi rep from its configured input source."""
        if getattr(self, f"{which}_head") is None:
            raise ValueError(f"the {which} head is disabled (enable_psi_head=False)")
        source = getattr(self, f"{which}_input")
        hidden, mask = (prefix_hidden, prefix_mask) if source == "state" else (suffix_hidden, suffix_mask)
        if hidden is None:
            seq = "prefix" if source == "state" else "suffix"
            raise ValueError(f"{which}_input={source!r} requires {seq} hidden states")
        mem = self._mix_layers(hidden, getattr(self, f"{which}_mix").value, self.rep_backbone_grad_scale)
        head = getattr(self, f"{which}_head")
        proj = getattr(self, f"{which}_proj")
        return proj(head(mem, mask, deterministic=deterministic, dropout_rng=dropout_rng))

    def _finalize_rep(self, which: Literal["phi", "psi"], rep):
        """Apply the rep's normalization policy (always float32)."""
        normalize = self._normalize_phi if which == "phi" else self._normalize_psi
        return _l2_normalize(rep) if normalize else rep.astype(jnp.float32)

    def _reps_from_prefix(self, observation: _model.Observation, which: tuple[str, ...]):
        """Preprocess + ONE prefix pass, then read the requested prefix-sourced reps from it.

        Every rep in `which` must have its input set to "state" (a prefix pass has no
        action tokens). Returns (reps_tuple, kv_cache, prefix_mask, prefix_len).
        """
        for w in which:
            if getattr(self, f"{w}_input") != "state":
                raise ValueError(f"computing {w} from a prefix-only pass requires {w}_input='state'")
        observation = _model.preprocess_observation(None, observation, train=False)
        _, prefix_hidden, kv_cache, prefix_mask, prefix_len = self._run_prefix(observation)
        reps = tuple(self._finalize_rep(w, self._compute_rep(w, prefix_hidden, None, prefix_mask, None)) for w in which)
        return reps, kv_cache, prefix_mask, prefix_len

    def get_state_representations(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions | None = None,
        timestep: at.Float[at.Array, "b"] | None = None,
        z: at.Float[at.Array, "b d"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b emb"] | None,
        at.Float[at.Array, "b emb"] | None,
        "KVCache",
        at.Bool[at.Array, "b s"],
        int,
    ]:
        """Unified state-side accessor: every rep computable from a full forward pass.

        Reps whose head is disabled (enable_psi_head=False) come back None. A
        state_action-sourced phi needs noisy_actions and timestep for the suffix pass;
        without them the phi slot is None.
        """
        which = tuple(
            w
            for w in ("psi", "phi")
            if getattr(self, f"{w}_input") == "state" and getattr(self, f"{w}_head") is not None
        )
        reps, kv_cache, prefix_mask, prefix_len = self._reps_from_prefix(observation, which)
        by_name = dict(zip(which, reps, strict=True))
        psi = by_name.get("psi")
        phi = by_name.get("phi")
        if phi is None and self.phi_input == "state_action" and noisy_actions is not None and timestep is not None:
            phi = self.get_phi_representation(
                observation, noisy_actions, timestep, kv_cache, prefix_mask, prefix_len, z=z
            )[0]
        return psi, phi, kv_cache, prefix_mask, prefix_len

    def get_phi_representation(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "b"],
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_len: int,
        z: at.Float[at.Array, "b d"] | None = None,
    ) -> tuple[at.Float[at.Array, "*b emb"], at.Float[at.Array, "*b ah emb"], at.Float[at.Array, "*b ah ad"]]:
        """Compute the phi representation (and velocity) from a suffix pass.

        Requires phi_input="state_action" (with phi_input="state" use get_state_representations
        + compute_velocity_step instead). Returns (phi_rep, action_hidden, v_t).
        """
        if self.phi_input != "state_action":
            raise ValueError(
                "get_phi_representation requires phi_input='state_action' (suffix phi); "
                "with phi_input='state' use get_state_representations and compute_velocity_step."
            )
        suffix_out, suffix_hidden, suffix_mask = self._suffix_forward(
            observation, noisy_actions, timestep, kv_cache, prefix_mask, prefix_len, z=z
        )
        phi_rep = self._finalize_rep("phi", self._compute_rep("phi", None, suffix_hidden, None, suffix_mask))
        action_hidden = suffix_out[:, -self.action_horizon :]
        v_t = self.action_out_proj(action_hidden)
        return phi_rep, action_hidden, v_t
