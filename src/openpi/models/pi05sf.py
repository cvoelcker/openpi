"""TD Successor-Features variant of pi05.

NOTE on naming: this model uses CRL's (pi05crl.py) phi/psi convention — psi = state readout
(L2-normalized state feature), phi = successor readout. This is INVERTED relative to
successor-feature literature (where phi is the state feature and psi the successor). The
convention is kept consistent with pi05crl.py to ease a future CRL+SF merge.

Like Pi0CRL, the reps come from Pi0RepBase's dedicated attention-pooling heads over a learned
softmax mix of all backbone layers; there are no readout tokens in the backbone sequence, so
the backbone passes are pure pi05.
"""

import logging
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.rep_base import Pi0RepBase
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class Pi0SF(Pi0RepBase):
    # phi(s,a) is the successor feature — its magnitude matters, so it stays unnormalized.
    # psi(s) is L2-normalized (the anti-collapse mechanism).
    _normalize_phi = False
    _normalize_psi = True

    def __init__(self, config: pi0_config.Pi0SFConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.sf_loss_coeff = config.sf_loss_coeff
        self.sf_gamma = config.sf_gamma
        self.fb_train_goal_ratio = config.fb_train_goal_ratio
        # z conditions F via the action-expert AdaRMS pathway (keeps B=psi in the prefix z-free).
        # Note: with rep_backbone_grad_scale=0.0 the SF loss is cut off from the backbone, so
        # z_proj only receives gradient through the (grad-scaled) backbone path.
        self.z_proj = nnx.Linear(config.rep_dim, self.action_in_proj.out_features, rngs=rngs)

    @override
    def _adarms_cond(
        self, time_emb: at.Float[at.Array, "b emb"], z: at.Float[at.Array, "b d"] | None
    ) -> at.Float[at.Array, "b emb"]:
        return time_emb if z is None else time_emb + self.z_proj(z)

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

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        batch: dict[str, Any],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """Semi-gradient SARSA TD successor-feature loss + flow-matching action loss.

        Required batch keys: observation, future_observation, actions, next_observation,
        next_actions (SARSA a'). Optional: next_is_pad (terminal masking of the bootstrap).
        """
        _model.require_batch_keys(
            batch,
            ("observation", "future_observation", "actions", "next_observation", "next_actions"),
            type(self).__name__,
        )
        observation = batch["observation"]
        future_observation = batch["future_observation"]
        actions = batch["actions"]
        next_observation = batch["next_observation"]
        next_actions = batch["next_actions"]
        next_is_pad = batch.get("next_is_pad")

        preprocess_rng, next_preprocess_rng, noise_rng, time_rng, z_rng, dropout_rng = jax.random.split(rng, 6)
        # Independent head-dropout keys per rep (only used when train=True).
        phi_drop_rng, psi_drop_rng = jax.random.split(dropout_rng)
        deterministic = not train
        # Augment current/next independently (consistent with pi05crl) so phi(s,a) and the
        # bootstrap phi(s',a') cannot match on shared augmentation artifacts.
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        next_observation = _model.preprocess_observation(next_preprocess_rng, next_observation, train=train)

        # Task latent z (detached): B(future) for a fraction of the batch, random for the rest.
        # z = B(future): use the already-forwarded future_observation (no train-loop change).
        psi_goal = self.get_state_representations(future_observation)[0]
        if psi_goal is None:
            raise ValueError("Pi0SF's z = B(future) requires psi_input='state'")
        z_goal = jax.lax.stop_gradient(psi_goal)
        z, z_goal_frac = self.sample_mixed_z(z_goal, z_rng)
        z_stacked = jnp.concatenate([z, z], axis=0)  # same z for the [obs, next] stack halves

        batch_size = actions.shape[0]

        # ---- (1) Flow-matching action loss on the current observation (standard pi05) ----
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        suffix_out, *_ = self._forward(observation, x_t, time)
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # ---- (2) SF pass: clean actions (time=0 -> x_t == actions), stacked current+next ----
        obs_stacked = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), observation, next_observation
        )
        actions_stacked = jnp.concatenate([actions, next_actions], axis=0)
        time0 = jnp.zeros((actions_stacked.shape[0],), dtype=jnp.float32)  # clean: x_t == actions
        _, prefix_hidden, suffix_hidden, prefix_mask, suffix_mask = self._forward(
            obs_stacked, actions_stacked, time0, z=z_stacked
        )

        # Reps via Pi0RepBase._compute_rep (input source per config; default psi(s) from the
        # prefix, phi(s,a) from the suffix). Layer axis is 0, so slice the batch on axis 1.
        def _rep_slice(which, sl, drop_rng, *, det):
            return self._compute_rep(
                which,
                prefix_hidden[:, sl],
                suffix_hidden[:, sl],
                prefix_mask[sl],
                suffix_mask[sl],
                deterministic=det,
                dropout_rng=drop_rng,
            )

        cur = slice(None, batch_size)
        nxt = slice(batch_size, None)
        # psi(s): CURRENT half, L2-normalized (anti-collapse).
        psi = self._finalize_rep("psi", _rep_slice("psi", cur, psi_drop_rng, det=deterministic))
        # phi(s,a): current half carries grad (and train-time dropout); the next half is the
        # stop-grad bootstrap and is computed deterministically — a noisy TD target would
        # inject variance into the regression.
        phi_sa = _rep_slice("phi", cur, phi_drop_rng, det=deterministic).astype(jnp.float32)
        phi_next = jax.lax.stop_gradient(_rep_slice("phi", nxt, None, det=True).astype(jnp.float32))

        # Mask the bootstrap at episode boundaries (terminal): target = psi(s) only.
        if next_is_pad is None:
            not_terminal = jnp.ones((batch_size, 1), dtype=phi_next.dtype)
        else:
            not_terminal = (1.0 - next_is_pad.astype(phi_next.dtype))[:, None]
        target = psi + self.sf_gamma * not_terminal * phi_next
        td_resid = phi_sa - target
        sf_loss = jnp.mean(jnp.square(td_resid))

        # Diagnostics (meaned by the train loop); layer-mix entropy watches for mix collapse.
        phi_mix_w = jax.nn.softmax(self.phi_mix.value.astype(jnp.float32))
        psi_mix_w = jax.nn.softmax(self.psi_mix.value.astype(jnp.float32))
        info = {
            "action_loss": jnp.mean(action_loss),
            "sf_loss": sf_loss,
            "psi_norm": jnp.mean(jnp.linalg.norm(psi, axis=-1)),
            "phi_norm": jnp.mean(jnp.linalg.norm(phi_sa, axis=-1)),
            "sf_td_resid": jnp.mean(jnp.linalg.norm(td_resid, axis=-1)),
            "z_goal_frac": z_goal_frac,
            "phi_mix_entropy": -jnp.sum(phi_mix_w * jnp.log(phi_mix_w + 1e-6)),
            "psi_mix_entropy": -jnp.sum(psi_mix_w * jnp.log(psi_mix_w + 1e-6)),
            "phi_mix_max": jnp.max(phi_mix_w),
            "psi_mix_max": jnp.max(psi_mix_w),
        }
        total_loss = self.action_loss_coeff * action_loss + self.sf_loss_coeff * sf_loss
        # "rep_loss" aliases sf_loss so the train loop's shared logging (which expects the CRL
        # model's key set) works unchanged.
        return total_loss, {"rep_loss": sf_loss, **info}
