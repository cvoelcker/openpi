from __future__ import annotations

import dataclasses
import logging

import einops
import flax
import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def _feature_dropout(x, rate: float, *, deterministic: bool, rng):
    """Inverted dropout with an explicitly-passed jax key (avoids the flax rng-collection
    machinery, which does not thread cleanly through the nnx bridge). No-op unless training."""
    if rate <= 0.0 or deterministic or rng is None:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep, x.shape)
    return jnp.where(mask, x / keep, 0.0)


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
        attn_mask = make_attn_mask(seq_mask, ar_mask)[:, None]  # (b, 1, t+1, t+1)
        positions = jnp.cumsum(seq_mask, axis=1) - 1
        for i in range(self.config.depth):
            (seq,), _ = _gemma.Block(configs=(self.config,), name=f"layers_{i}")(
                [seq], None, positions, attn_mask, [None]
            )
        seq, _ = _gemma.RMSNorm(name="final_norm")(seq, None)
        pooled = seq[:, -1]  # (b, d) — the pooled query row
        return _feature_dropout(pooled, self.dropout, deterministic=deterministic, rng=dropout_rng)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.crl_loss_coeff = config.crl_loss_coeff
        self.action_loss_coeff = config.action_loss_coeff
        # Training input of each CRL rep. psi (future-time target) is always
        # state-input; phi (current-time anchor) is configurable.
        self.phi_input = config.phi_input
        self.psi_input = "state"
        self._phi_on_prefix = config.phi_input == "state"
        # Both reps project to rep_dim, so the critic conditioning size is the
        # same regardless of which rep is selected downstream.
        self.phi_dim = config.rep_dim
        self.psi_dim = config.rep_dim
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
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                return_hidden=True,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.rep_backbone_grad_scale = config.rep_backbone_grad_scale
        num_backbone_layers = paligemma_config.depth

        psi_mem_width = paligemma_config.width
        phi_mem_width = paligemma_config.width if self._phi_on_prefix else action_expert_config.width

        def _head_config(width):
            return dataclasses.replace(paligemma_config, width=width, depth=config.rep_head_depth, lora_configs={})

        self.rep_head_dropout = config.rep_head_dropout
        psi_head = nnx_bridge.ToNNX(_RepHead(_head_config(psi_mem_width), config.dtype, dropout=config.rep_head_dropout))
        psi_head.lazy_init(jnp.zeros((1, 2, psi_mem_width)), jnp.ones((1, 2), dtype=bool), rngs=rngs)
        phi_head = nnx_bridge.ToNNX(_RepHead(_head_config(phi_mem_width), config.dtype, dropout=config.rep_head_dropout))
        phi_head.lazy_init(jnp.zeros((1, 2, phi_mem_width)), jnp.ones((1, 2), dtype=bool), rngs=rngs)
        self.psi_head = psi_head
        self.phi_head = phi_head

        # Learned softmax mix over all backbone layers, initialized to favor early layers.
        self.psi_mix = nnx.Param(jnp.linspace(1.0, -1.0, num_backbone_layers))
        self.phi_mix = nnx.Param(jnp.linspace(1.0, -1.0, num_backbone_layers))

        self.phi_proj = nnx.Linear(phi_mem_width, config.rep_dim, rngs=rngs)
        self.psi_proj = nnx.Linear(psi_mem_width, config.rep_dim, rngs=rngs)

        # Learnable CRL temperature. Reps are L2-normalized before the InfoNCE dot products,
        # so logits are cosine sims; logit_scale = log(1/temperature) rescales them (CLIP-style),
        # clamped at exp(logit_scale) <= 100 to avoid runaway. Named to match the freeze filter.
        self.logit_scale = nnx.Param(jnp.asarray(jnp.log(1.0 / config.crl_temperature_init), dtype=jnp.float32))
        self.logsumexp_penalty_coeff = config.logsumexp_penalty_coeff

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

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

    def _psi_from_prefix(self, prefix_hidden, prefix_mask, *, deterministic=True, dropout_rng=None):
        mem = self._mix_layers(prefix_hidden, self.psi_mix.value, self.rep_backbone_grad_scale)
        return self.psi_proj(self.psi_head(mem, prefix_mask, deterministic=deterministic, dropout_rng=dropout_rng))

    def _phi_from_prefix(self, prefix_hidden, prefix_mask, *, deterministic=True, dropout_rng=None):
        mem = self._mix_layers(prefix_hidden, self.phi_mix.value, self.rep_backbone_grad_scale)
        return self.phi_proj(self.phi_head(mem, prefix_mask, deterministic=deterministic, dropout_rng=dropout_rng))

    def _phi_from_suffix(self, suffix_hidden, suffix_mask, *, deterministic=True, dropout_rng=None):
        mem = self._mix_layers(suffix_hidden, self.phi_mix.value, self.rep_backbone_grad_scale)
        return self.phi_proj(self.phi_head(mem, suffix_mask, deterministic=deterministic, dropout_rng=dropout_rng))

    def _run_prefix(self, observation):
        """Prefix-only backbone pass. Returns (prefix_out, prefix_hidden, kv_cache, prefix_mask, prefix_len)."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache, (prefix_hidden, _) = self.PaliGemma.llm(
            [prefix_tokens, None], mask=attn_mask, positions=positions
        )
        return prefix_out, prefix_hidden, kv_cache, prefix_mask, prefix_tokens.shape[1]

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _suffix_forward(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "b"],
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_len: int,
    ):
        """Shared helper: embed suffix, build prefix+suffix attention mask, run the action expert."""
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, noisy_actions, timestep
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (_, suffix_out), _, _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out

    def get_psi_representation(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b emb"], "KVCache", at.Bool[at.Array, "b s"], int]:
        """Compute the action-independent (psi) representation.

        Returns (psi_rep, kv_cache, prefix_mask, prefix_len) where psi_rep is
        the psi head's attention-pooled readout over the layer-mixed prefix.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        _, prefix_hidden, kv_cache, prefix_mask, prefix_len = self._run_prefix(observation)
        psi_rep = self._psi_from_prefix(prefix_hidden, prefix_mask)
        return psi_rep, kv_cache, prefix_mask, prefix_len

    def get_state_representations(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b emb"], at.Float[at.Array, "b emb"], "KVCache", at.Bool[at.Array, "b s"], int]:
        """State-only mode: compute both representations from a single prefix pass.

        Returns (psi_rep, phi_rep, kv_cache, prefix_mask, prefix_len).
        Both reps are action-independent; use this instead of
        get_phi_representation when phi_input="state".
        """
        if not self._phi_on_prefix:
            raise ValueError("get_state_representations requires phi_input='state'")
        observation = _model.preprocess_observation(None, observation, train=False)
        _, prefix_hidden, kv_cache, prefix_mask, prefix_len = self._run_prefix(observation)
        psi_rep = self._psi_from_prefix(prefix_hidden, prefix_mask)
        phi_rep = self._phi_from_prefix(prefix_hidden, prefix_mask)
        return psi_rep, phi_rep, kv_cache, prefix_mask, prefix_len

    def get_phi_representation(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, "b"],
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_len: int,
    ) -> tuple[at.Float[at.Array, "*b emb"], at.Float[at.Array, "*b ah emb"], at.Float[at.Array, "*b ah ad"]]:
        """Compute the action-dependent (phi) representation and velocity.

        Returns (phi_rep, action_hidden, v_t) where phi_rep is the phi head's
        attention-pooled readout over the layer-mixed suffix, action_hidden is
        the transformer hidden states for action tokens, and v_t is the velocity.

        Not available with phi_input="state" (phi is not part of the suffix there);
        use get_state_representations + compute_velocity_step instead.
        """
        if self._phi_on_prefix:
            raise ValueError(
                "get_phi_representation requires phi_input='state_action' (suffix phi); "
                "with phi_input='state' use get_state_representations and compute_velocity_step."
            )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, noisy_actions, timestep
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _, (_, suffix_hidden) = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        phi_rep = self._phi_from_suffix(suffix_hidden, suffix_mask)
        action_hidden = suffix_out[:, -self.action_horizon :]
        v_t = self.action_out_proj(action_hidden)
        return phi_rep, action_hidden, v_t

    def get_prefix_cache(
        self, observation: _model.Observation
    ) -> tuple["KVCache", at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"]]:
        """Compatibility wrapper: returns (kv_cache, prefix_hidden, prefix_mask)."""
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_out, _, kv_cache, prefix_mask, _ = self._run_prefix(observation)
        return kv_cache, prefix_out, prefix_mask

    def compute_velocity_step(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        t_pi0: float | at.Float[at.Array, " b"],
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b s"],
    ) -> at.Float[at.Array, "b ah ad"]:
        """Compatibility wrapper: compute velocity at a single ODE step."""
        batch_size = x_t.shape[0]
        timestep = jnp.broadcast_to(jnp.asarray(t_pi0), (batch_size,))
        prefix_len = prefix_mask.shape[1]
        suffix_out = self._suffix_forward(observation, x_t, timestep, kv_cache, prefix_mask, prefix_len)
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        future_observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        episode_id: at.Int[at.Array, " b"] | None = None,
        negative_observation: _model.Observation | None = None,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, future_preprocess_rng, negative_preprocess_rng, noise_rng, time_rng, dropout_rng = (
            jax.random.split(rng, 6)
        )
        # Independent head-dropout keys per rep (only used when train=True).
        phi_drop_rng, psi_drop_rng, neg_drop_rng = jax.random.split(dropout_rng, 3)
        deterministic = not train
        # Augment each contrastive view independently — a shared key would apply the
        # identical crop/rotation/jitter to each anchor/future pair, letting the model match
        # on shared appearance artifacts instead of learning temporal state progress.
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        future_observation = _model.preprocess_observation(future_preprocess_rng, future_observation, train=train)

        # Stack current + future (+ optional explicit within-task negative) for one forward pass.
        # Actions are duplicated for the future/negative halves — those suffixes are discarded, so
        # the duplicated actions never enter the action loss.
        obs_group = [observation, future_observation]
        if negative_observation is not None:
            negative_observation = _model.preprocess_observation(
                negative_preprocess_rng, negative_observation, train=train
            )
            obs_group.append(negative_observation)
        observation_stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *obs_group
        )
        actions_stacked = jnp.concatenate([actions] * len(obs_group), axis=0)

        orig_batch_size = actions.shape[0]
        noise = jax.random.normal(noise_rng, actions_stacked.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (actions_stacked.shape[0],)) * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions_stacked
        u_t = noise - actions_stacked

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation_stacked)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation_stacked, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _, (prefix_hidden, suffix_hidden) = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        # Action loss: current-obs half only.
        v_t = self.action_out_proj(suffix_out[:orig_batch_size, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t[:orig_batch_size]), axis=-1)

        # CRL reps via dedicated heads over a learned layer-mix. psi always reads a prefix
        # (state-only); phi reads the current prefix (state mode) or current suffix
        # (state_action mode). Layer axis is 0, so slice batch on axis 1. The stack is
        # [current | future | (negative)], so slices are [:B], [B:2B], [2B:3B].
        B = orig_batch_size
        psi_raw = self._psi_from_prefix(
            prefix_hidden[:, B : 2 * B], prefix_mask[B : 2 * B], deterministic=deterministic, dropout_rng=psi_drop_rng
        )  # (B, d)
        if self._phi_on_prefix:
            phi_raw = self._phi_from_prefix(
                prefix_hidden[:, :B], prefix_mask[:B], deterministic=deterministic, dropout_rng=phi_drop_rng
            )  # (B, d)
        else:
            phi_raw = self._phi_from_suffix(
                suffix_hidden[:, :B], suffix_mask[:B], deterministic=deterministic, dropout_rng=phi_drop_rng
            )  # (B, d)

        # L2-normalize reps onto the unit sphere so InfoNCE logits are cosine similarities.
        # This removes the norm-inflation shortcut (driving train loss down by growing ||rep||
        # rather than improving alignment) and forces the heads to separate directions. Cast to
        # float32 first — reps come out of the projection in the (bf16) embed dtype, and the
        # normalization + logit/logsumexp math wants full precision.
        eps = 1e-6
        phi_raw = phi_raw.astype(jnp.float32)
        psi_raw = psi_raw.astype(jnp.float32)
        phi_flat = phi_raw / (jnp.linalg.norm(phi_raw, axis=-1, keepdims=True) + eps)
        psi_flat = psi_raw / (jnp.linalg.norm(psi_raw, axis=-1, keepdims=True) + eps)
        # Learnable temperature (clamped) restores separability of the bounded cosine logits.
        logit_scale = jnp.exp(jnp.minimum(self.logit_scale.value, jnp.log(100.0)))

        crl_matrix = logit_scale * jnp.einsum("id,jd->ij", psi_flat, phi_flat)  # (B, B): [i, j] = <psi_i, phi_j>
        crl_pos = jnp.diag(crl_matrix)

        # Explicit within-task negative: an extra candidate future (psi-type) for its own anchor.
        # It enters only the anchor->future direction (normalize over psi), as an extra key per
        # anchor column j; a negative goal is not the positive of any anchor, so it has no query
        # role in the future->anchor direction. Mixing this within-task hard negative with the
        # (mostly cross-task) in-batch negatives keeps the InfoNCE denominator a valid mixture
        # reference while forcing phi/psi to discriminate states *within* a task.
        if negative_observation is not None:
            psi_neg_raw = self._psi_from_prefix(
                prefix_hidden[:, 2 * B : 3 * B],
                prefix_mask[2 * B : 3 * B],
                deterministic=deterministic,
                dropout_rng=neg_drop_rng,
            )  # (B, d)
            psi_neg_raw = psi_neg_raw.astype(jnp.float32)
            psi_neg = psi_neg_raw / (jnp.linalg.norm(psi_neg_raw, axis=-1, keepdims=True) + eps)
            crl_self_neg = logit_scale * jnp.sum(phi_flat * psi_neg, axis=-1)  # (B,): [j] = <phi_j, psi_neg_j>
            over_psi_logits = jnp.concatenate([crl_matrix, crl_self_neg[None, :]], axis=0)  # (B+1, B)
        else:
            crl_self_neg = None
            over_psi_logits = crl_matrix

        # Symmetric InfoNCE: contrast over anchors (axis=1) AND over futures (axis=0).
        crl_neg_over_phi = jax.nn.logsumexp(crl_matrix, axis=1)  # future psi_i, normalize over phi anchors
        crl_neg_over_psi = jax.nn.logsumexp(over_psi_logits, axis=0)  # anchor phi_j, normalize over psi (+ its neg)
        crl_infonce = 0.5 * jnp.mean((crl_neg_over_phi - crl_pos) + (crl_neg_over_psi - crl_pos))

        # Logsumexp penalty (guards against logit blow-up / collapse); coeff is a config knob.
        logsumexp_penalty = 0.5 * (jnp.mean(crl_neg_over_phi**2) + jnp.mean(crl_neg_over_psi**2))
        crl_loss = crl_infonce + self.logsumexp_penalty_coeff * logsumexp_penalty

        # Diagnostics (meaned by the train loop). Pre-normalization rep norms confirm whether the
        # heads were inflating norms; layer-mix entropy watches for mix collapse; collision_rate is
        # the fraction of off-diagonal in-batch pairs sharing an episode (false-negative pressure).
        phi_mix_w = jax.nn.softmax(self.phi_mix.value.astype(jnp.float32))
        psi_mix_w = jax.nn.softmax(self.psi_mix.value.astype(jnp.float32))
        diagnostics = {
            "phi_norm": jnp.mean(jnp.linalg.norm(phi_raw, axis=-1)),
            "psi_norm": jnp.mean(jnp.linalg.norm(psi_raw, axis=-1)),
            "temperature": 1.0 / logit_scale,
            "phi_mix_entropy": -jnp.sum(phi_mix_w * jnp.log(phi_mix_w + eps)),
            "psi_mix_entropy": -jnp.sum(psi_mix_w * jnp.log(psi_mix_w + eps)),
            "phi_mix_max": jnp.max(phi_mix_w),
            "psi_mix_max": jnp.max(psi_mix_w),
        }
        if crl_self_neg is not None:
            # Within-task discrimination readout (compute-only, NOT in total_loss): a pure 2-way
            # InfoNCE contrasting each anchor's true future (crl_pos[j] = <psi_j, phi_j>) against
            # its own within-task hard negative (crl_self_neg[j] = <phi_j, psi_neg_j>). Both share
            # anchor phi_j, so this isolates whether the head separates *states within a task* —
            # the shortcut-proof signal (prompt/scene fingerprints cancel). Bounded by ln(2)≈0.693
            # (chance for 2 candidates); →0 = perfect within-task discrimination.
            diagnostics["within_task_loss"] = jnp.mean(jnp.logaddexp(crl_pos, crl_self_neg) - crl_pos)
        if episode_id is not None:
            same_ep = episode_id[:, None] == episode_id[None, :]  # (B, B)
            off_diag = ~jnp.eye(B, dtype=jnp.bool_)
            n_off = jnp.sum(off_diag)
            diagnostics["collision_rate"] = jnp.sum(same_ep & off_diag) / jnp.maximum(n_off, 1)

            # Compute-only confound probe (NOT in total_loss, no backprop): recompute the InfoNCE
            # term with same-episode off-diagonal pairs removed from the denominator. These pairs
            # are false negatives (genuinely similar frames penalized as negatives); val batches
            # carry ~14x more of them than train (tiny val episode pool), so if the train/val gap
            # is a collision artifact rather than memorization, rep_loss_masked closes it while
            # rep_loss_nomask reproduces the gap. Diagonal (positive) is never masked.
            false_neg = same_ep & off_diag  # (B, B)
            masked_crl = jnp.where(false_neg, -1e9, crl_matrix)
            if negative_observation is not None:
                masked_over_psi = jnp.concatenate([masked_crl, crl_self_neg[None, :]], axis=0)
            else:
                masked_over_psi = masked_crl
            masked_neg_over_phi = jax.nn.logsumexp(masked_crl, axis=1)
            masked_neg_over_psi = jax.nn.logsumexp(masked_over_psi, axis=0)
            diagnostics["rep_loss_masked"] = 0.5 * jnp.mean(
                (masked_neg_over_phi - crl_pos) + (masked_neg_over_psi - crl_pos)
            )
            # Unmasked InfoNCE (no penalty) as the apples-to-apples reference for rep_loss_masked.
            diagnostics["rep_loss_nomask"] = crl_infonce

        total_loss = self.action_loss_coeff * action_loss + self.crl_loss_coeff * crl_loss
        return total_loss, {"action_loss": action_loss, "rep_loss": crl_loss, **diagnostics}

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        prefix_len = prefix_tokens.shape[1]

        def step(carry):
            x_t, time = carry
            suffix_out = self._suffix_forward(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                kv_cache,
                prefix_mask,
                prefix_len,
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
