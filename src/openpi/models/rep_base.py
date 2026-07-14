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


class _RepHead(nn.Module):
    """Attention-pooling head: reuses gemma Blocks to pool a layer-mixed backbone
    token sequence into a single vector via a learned query token."""

    config: _gemma.Config
    embed_dtype: str
    dropout: float = 0.0

    @nn.compact
    def __call__(self, memory, memory_mask, *, deterministic: bool = True, dropout_rng=None):
        # memory: (b, t, d); memory_mask: (b, t) bool
        memory = memory.astype(self.embed_dtype)
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

        def _head_config(width):
            return dataclasses.replace(paligemma_config, width=width, depth=config.rep_head_depth, lora_configs={})

        psi_head = nnx_bridge.ToNNX(
            _RepHead(_head_config(psi_mem_width), config.dtype, dropout=config.rep_head_dropout)
        )
        psi_head.lazy_init(jnp.zeros((1, 2, psi_mem_width)), jnp.ones((1, 2), dtype=bool), rngs=rngs)
        phi_head = nnx_bridge.ToNNX(
            _RepHead(_head_config(phi_mem_width), config.dtype, dropout=config.rep_head_dropout)
        )
        phi_head.lazy_init(jnp.zeros((1, 2, phi_mem_width)), jnp.ones((1, 2), dtype=bool), rngs=rngs)
        self.psi_head = psi_head
        self.phi_head = phi_head

        # Learned softmax mix over all backbone layers, initialized to favor early layers.
        self.psi_mix = nnx.Param(jnp.linspace(1.0, -1.0, num_backbone_layers))
        self.phi_mix = nnx.Param(jnp.linspace(1.0, -1.0, num_backbone_layers))

        self.phi_proj = nnx.Linear(phi_mem_width, config.rep_dim, rngs=rngs)
        self.psi_proj = nnx.Linear(psi_mem_width, config.rep_dim, rngs=rngs)

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

    def get_psi_representation(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b emb"], "KVCache", at.Bool[at.Array, "b s"], int]:
        """Compute the psi representation from a prefix-only pass.

        Requires psi_input="state" (a prefix pass has no action tokens).
        Returns (psi_rep, kv_cache, prefix_mask, prefix_len).
        """
        if self.psi_input != "state":
            raise ValueError("get_psi_representation requires psi_input='state'")
        observation = _model.preprocess_observation(None, observation, train=False)
        _, prefix_hidden, kv_cache, prefix_mask, prefix_len = self._run_prefix(observation)
        psi_rep = self._finalize_rep("psi", self._compute_rep("psi", prefix_hidden, None, prefix_mask, None))
        return psi_rep, kv_cache, prefix_mask, prefix_len

    def get_state_representations(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b emb"], at.Float[at.Array, "b emb"], "KVCache", at.Bool[at.Array, "b s"], int]:
        """State-only mode: compute both representations from a single prefix pass.

        Requires phi_input="state" and psi_input="state" (both reps action-independent).
        Returns (psi_rep, phi_rep, kv_cache, prefix_mask, prefix_len).
        """
        if self.phi_input != "state" or self.psi_input != "state":
            raise ValueError("get_state_representations requires phi_input='state' and psi_input='state'")
        observation = _model.preprocess_observation(None, observation, train=False)
        _, prefix_hidden, kv_cache, prefix_mask, prefix_len = self._run_prefix(observation)
        psi_rep = self._finalize_rep("psi", self._compute_rep("psi", prefix_hidden, None, prefix_mask, None))
        phi_rep = self._finalize_rep("phi", self._compute_rep("phi", prefix_hidden, None, prefix_mask, None))
        return psi_rep, phi_rep, kv_cache, prefix_mask, prefix_len

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
