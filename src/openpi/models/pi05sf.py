import logging

import einops
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


class Pi0SF(_model.BaseModel):
    """TD Successor-Features variant of pi05.

    NOTE on naming: this model uses CRL's (pi05rep.py) phi/psi convention — psi = prefix/state
    token (L2-normalized state feature), phi = suffix/successor token. This is INVERTED relative
    to successor-feature literature (where phi is the state feature and psi the successor). The
    convention is kept consistent with pi05rep.py to ease a future CRL+SF merge.
    """

    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
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

        # Successor-features readout, using CRL's phi/psi convention (see class header):
        #   psi(s): state-feature token in the PREFIX (paligemma width), projected to sf_dim and
        #     L2-normalized — the anti-collapse mechanism.
        #   phi(s,a): successor token in the SUFFIX (action-expert width), projected to sf_dim.
        sf_dim = config.sf_dim
        self.sf_gamma = config.sf_gamma
        self.fb_train_goal_ratio = config.fb_train_goal_ratio
        self.psi_token = nnx.Param(
            jax.random.uniform(rngs.params(), (1, paligemma_config.width)) / paligemma_config.width
        )
        self.psi_proj = nnx.Linear(paligemma_config.width, sf_dim, rngs=rngs)
        self.phi_token = nnx.Param(
            jax.random.uniform(rngs.params(), (1, action_expert_config.width)) / action_expert_config.width
        )
        self.phi_proj = nnx.Linear(action_expert_config.width, sf_dim, rngs=rngs)
        # z conditions F via the action-expert AdaRMS pathway (keeps B=psi in the prefix z-free).
        self.z_proj = nnx.Linear(sf_dim, action_expert_config.width, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

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
        tokens.append(
            jnp.broadcast_to(self.psi_token.value[None], (obs.state.shape[0], 1, self.psi_token.value.shape[-1]))
        )
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # psi starts its own block: it attends to image/language, but they must not
        # attend forward into it (prevents representation leakage into the backbone).
        ar_mask += [True]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        z: at.Float[at.Array, "b d"] | None = None,
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
            adarms_cond = time_emb if z is None else time_emb + self.z_proj(z)
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
        tokens.append(
            jnp.broadcast_to(self.phi_token.value[None], (obs.state.shape[0], 1, self.phi_token.value.shape[-1]))
        )
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens or phi; phi attends to everything in the suffix
        ar_mask += [True] + ([False] * (self.action_horizon - 1)) + [True]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @property
    def _num_prefix_rep_tokens(self) -> int:
        """Number of rep tokens appended to the prefix. SF keeps psi on the prefix and phi in the
        suffix, so this is always 1 (mirrors pi05rep.py's helper for structural parity)."""
        return 1

    def _prefix_attn_mask(self, prefix_mask, prefix_ar_mask):
        """Prefix self-attention mask. SF has a single prefix rep token (psi), so this is just
        make_attn_mask; kept as a method to mirror pi05rep.py."""
        return make_attn_mask(prefix_mask, prefix_ar_mask)

    def _forward(self, observation, x_t, time, z=None):
        """Single PaliGemma forward over an observation + (noisy or clean) action chunk.

        Returns (prefix_out, suffix_out). The last suffix token is phi(s,a) (successor); the last
        prefix token is the psi(s) state-feature token. Suffix tokens are blocked from attending to
        the psi_token (last prefix token) to prevent representation leakage into the policy.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time, z=z)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        prefix_len = prefix_tokens.shape[1]
        attn_mask = attn_mask.at[:, prefix_len:, prefix_len - self._num_prefix_rep_tokens : prefix_len].set(False)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        return prefix_out, suffix_out

    def _suffix_forward(self, observation, noisy_actions, timestep, kv_cache, prefix_mask, prefix_len, z=None):
        """Suffix-only forward: embed suffix, block attention to the psi prefix token, run LLM."""
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, noisy_actions, timestep, z=z
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        prefix_attn_mask = prefix_attn_mask.at[:, :, prefix_len - self._num_prefix_rep_tokens : prefix_len].set(False)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return suffix_out

    def get_psi_representation(self, observation: _model.Observation):
        """Action-independent state feature psi(s): projected + L2-normalized last prefix token.

        Returns (psi, kv_cache, prefix_mask, prefix_len).
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = self._prefix_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        psi = self.psi_proj(prefix_out[:, -self._num_prefix_rep_tokens])
        psi = psi / (jnp.linalg.norm(psi, axis=-1, keepdims=True) + 1e-6)
        return psi, kv_cache, prefix_mask, prefix_tokens.shape[1]

    def get_phi_representation(
        self, observation, noisy_actions, timestep, kv_cache, prefix_mask, prefix_len, z=None
    ):
        """Action-dependent successor phi(s,a,z): projected last suffix token.

        Returns (phi, action_hidden, v_t).
        """
        suffix_out = self._suffix_forward(observation, noisy_actions, timestep, kv_cache, prefix_mask, prefix_len, z=z)
        phi = self.phi_proj(suffix_out[:, -1])
        action_hidden = suffix_out[:, : self.action_horizon]
        v_t = self.action_out_proj(action_hidden)
        return phi, action_hidden, v_t

    def get_prefix_cache(self, observation: _model.Observation):
        """Compatibility wrapper: returns (kv_cache, prefix_hidden, prefix_mask)."""
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = self._prefix_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        return kv_cache, prefix_out, prefix_mask

    def compute_velocity_step(self, observation, x_t, t_pi0, kv_cache, prefix_mask):
        """Compatibility wrapper: compute velocity at a single ODE step."""
        batch_size = x_t.shape[0]
        timestep = jnp.broadcast_to(jnp.asarray(t_pi0), (batch_size,))
        prefix_len = prefix_mask.shape[1]
        suffix_out = self._suffix_forward(observation, x_t, timestep, kv_cache, prefix_mask, prefix_len)
        return self.action_out_proj(suffix_out[:, : self.action_horizon])

    def sample_mixed_z(self, z_goal, rng):
        """Mix goal-derived z (= B(future), detached) with random unit-sphere z.

        Returns (z, goal_frac) where with prob fb_train_goal_ratio z_i = z_goal_i, else random.
        """
        b, d = z_goal.shape
        rng_n, rng_m = jax.random.split(rng)
        z_rand = jax.random.normal(rng_n, (b, d))
        z_rand = z_rand / (jnp.linalg.norm(z_rand, axis=-1, keepdims=True) + 1e-6)
        use_goal = jax.random.uniform(rng_m, (b, 1)) < self.fb_train_goal_ratio
        z = jnp.where(use_goal, z_goal, z_rand)
        return z, jnp.mean(use_goal.astype(jnp.float32))

    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        future_observation: _model.Observation,  # accepted for trainer-signature parity; unused by SF
        actions: _model.Actions,
        *,
        next_observation: _model.Observation | None = None,
        next_actions: _model.Actions | None = None,
        next_is_pad: at.Bool[at.Array, " b"] | None = None,
        train: bool = False,
        **_unused,  # signature parity with pi05rep.py's CRL compute_loss
    ) -> tuple[at.Float[at.Array, "*b ah"], dict]:
        # z = B(future): use the already-forwarded future_observation (no train_rep.py change).
        if next_observation is None or next_actions is None:
            raise ValueError("Pi0SF.compute_loss requires next_observation and next_actions (SARSA a').")
        preprocess_rng, noise_rng, time_rng, z_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        next_observation = _model.preprocess_observation(preprocess_rng, next_observation, train=train)

        # Task latent z (detached): B(future) for a fraction of the batch, random for the rest.
        z_goal = jax.lax.stop_gradient(self.get_psi_representation(future_observation)[0])
        z, z_goal_frac = self.sample_mixed_z(z_goal, z_rng)
        z_stacked = jnp.concatenate([z, z], axis=0)  # same z for the [obs, next] stack halves

        batch_size = actions.shape[0]

        # ---- (1) Flow-matching action loss on the current observation (standard pi05) ----
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        _, suffix_out = self._forward(observation, x_t, time)
        v_t = self.action_out_proj(suffix_out[:, : self.action_horizon])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # ---- (2) SF pass: clean actions (time=0 -> x_t == actions), stacked current+next ----
        obs_stacked = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), observation, next_observation
        )
        actions_stacked = jnp.concatenate([actions, next_actions], axis=0)
        time0 = jnp.zeros((actions_stacked.shape[0],), dtype=jnp.float32)  # clean: x_t == actions
        prefix_out, sf_suffix_out = self._forward(obs_stacked, actions_stacked, time0, z=z_stacked)

        # psi(s): last prefix token of the CURRENT half, projected + L2-normalized (anti-collapse).
        psi = self.psi_proj(prefix_out[:batch_size, -self._num_prefix_rep_tokens])
        psi = psi / (jnp.linalg.norm(psi, axis=-1, keepdims=True) + 1e-6)
        # phi(s,a): last suffix token (successor); current half carries grad, next half is the
        # stop-grad bootstrap.
        phi = self.phi_proj(sf_suffix_out[:, -1])
        phi_sa = phi[:batch_size]
        phi_next = jax.lax.stop_gradient(phi[batch_size:])

        # Mask the bootstrap at episode boundaries (terminal): target = psi(s) only.
        if next_is_pad is None:
            not_terminal = jnp.ones((batch_size, 1), dtype=phi_next.dtype)
        else:
            not_terminal = (1.0 - next_is_pad.astype(phi_next.dtype))[:, None]
        target = psi + self.sf_gamma * not_terminal * phi_next
        td_resid = phi_sa - target
        sf_loss = jnp.mean(jnp.square(td_resid))

        info = {
            "action_loss": jnp.mean(action_loss),
            "sf_loss": sf_loss,
            "psi_norm": jnp.mean(jnp.linalg.norm(psi, axis=-1)),
            "phi_norm": jnp.mean(jnp.linalg.norm(phi_sa, axis=-1)),
            "sf_td_resid": jnp.mean(jnp.linalg.norm(td_resid, axis=-1)),
            "z_goal_frac": z_goal_frac,
        }
        return action_loss + sf_loss, info

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
        prefix_attn_mask = self._prefix_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # block suffix tokens from attending to psi_token (last prefix token)
            prefix_attn_mask = prefix_attn_mask.at[
                :, :, prefix_tokens.shape[1] - self._num_prefix_rep_tokens : prefix_tokens.shape[1]
            ].set(False)
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
