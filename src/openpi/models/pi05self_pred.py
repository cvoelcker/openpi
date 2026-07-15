"""Latent self-prediction (SP) variant of pi05.

Pi0SP trains the phi readout with a latent forward model: a BRO-style residual MLP
(ForwardProjHead) predicts the next-step rep phi(s') from (phi(s), a). The prediction is
trained with (1) an MSE regression onto the stop-gradient target phi(s') and (2) SigREP,
LeJEPA's SIGReg (sketched isotropic-Gaussian regularization): random 1D projections of the
phi embeddings are pushed toward N(0, 1) via the Epps-Pulley characteristic-function test.
An isotropic-Gaussian embedding distribution rules out the constant-rep (and any
low-dimensional) collapse that the regression alone admits — without negatives, a
temperature, or other contrastive machinery.

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


class BROBlock(nnx.Module):
    """A BRO-style residual block with a linear layer, layer norm, and GELU activation.
    The input is added to the output to form the final output. This is used in the ForwardProjHead
    to ease learning the identity function when the action is not informative.
    """

    def __init__(self, latent_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(latent_dim, rngs=rngs)
        self.fc2 = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        residual = x
        x = self.fc1(x)
        x = self.ln1(x)
        x = nnx.gelu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x + residual


class ForwardProjHead(nnx.Module):
    """The latent forward model: predicts phi(s') from the concatenated (phi(s), actions).

    The input is the concatenation of the state rep and the flattened action sequence; the
    output lives in the same rep_dim space as phi so it can be regressed / contrasted against
    phi(s'). BRO-style residual blocks ease learning the identity function when the action is
    not informative.
    """

    def __init__(self, config: pi0_config.Pi0SPConfig, rngs: nnx.Rngs):
        blocks = config.forward_proj_blocks
        input_dim = config.rep_dim + config.action_dim * config.action_horizon
        latent_dim = config.rep_dim
        self.fc1 = nnx.Linear(input_dim, latent_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(latent_dim, rngs=rngs)
        # String-keyed nnx.Dict, NOT a plain list: list entries get integer path keys in
        # the param tree, which breaks the "/"-joined flatten in weight_loaders._merge_params.
        self.num_blocks = blocks
        self.block_list = nnx.Dict({f"block_{i}": BROBlock(latent_dim, rngs) for i in range(blocks)})

    def __call__(self, x: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        x = self.fc1(x)
        x = self.ln1(x)
        x = nnx.gelu(x)
        for i in range(self.num_blocks):
            x = self.block_list[f"block_{i}"](x)
        return x


def _sigreg_epps_pulley(
    x: at.Float[at.Array, "n d"],
    rng: at.KeyArrayLike,
    *,
    num_slices: int,
    t_max: float,
    num_t: int,
) -> at.Float[at.Array, ""]:
    """LeJEPA's SIGReg loss on a batch of embeddings.

    Sketch: project x onto num_slices random unit directions (fresh every call, so the
    directions cover the sphere over training). Per direction, compute the Epps-Pulley
    statistic — the N(0,1)-weighted integral over t of |cf_emp(t) - exp(-t^2/2)|^2, where
    cf_emp is the empirical characteristic function of the n projected samples. The
    statistic is 0 iff every projection is standard normal, i.e. iff the embedding
    distribution is isotropic Gaussian (which also pins its scale, so x must NOT be
    L2-normalized). Smooth, bounded, and O(n * num_slices * num_t) — no pairwise terms.
    """
    x = x.astype(jnp.float32)
    dirs = jax.random.normal(rng, (num_slices, x.shape[-1]))
    dirs = dirs / (jnp.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-6)
    proj = x @ dirs.T  # (n, m)
    t = jnp.linspace(-t_max, t_max, num_t)
    tx = proj[..., None] * t  # (n, m, T)
    cf_real = jnp.mean(jnp.cos(tx), axis=0)  # (m, T)
    cf_imag = jnp.mean(jnp.sin(tx), axis=0)  # (m, T)
    gauss_cf = jnp.exp(-0.5 * t**2)
    sq_err = (cf_real - gauss_cf) ** 2 + cf_imag**2
    weight = jax.scipy.stats.norm.pdf(t)
    ep = jnp.trapezoid(sq_err * weight, t, axis=-1)  # (m,)
    return jnp.mean(ep)


class Pi0SP(Pi0RepBase):
    # phi must stay unnormalized: SIGReg drives its distribution to an isotropic Gaussian,
    # which pins the scale — L2 normalization would fight the regularizer (and destroy the
    # Gaussian shape). There is no psi head (Pi0SPConfig sets enable_psi_head=False); psi
    # accessors serve phi.
    _normalize_phi = False

    def __init__(self, config: pi0_config.Pi0SPConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.sp_loss_coeff = config.sp_loss_coeff
        self.sigrep_loss_coeff = config.sigrep_loss_coeff
        # z conditions the action expert via the AdaRMS pathway (keeps the prefix z-free).
        # Unused by compute_loss (no z is passed); kept for inference-time steering.
        self.z_proj = nnx.Linear(config.rep_dim, self.action_in_proj.out_features, rngs=rngs)

        self.forward_proj = ForwardProjHead(config, rngs)

        # SIGReg sketch/quadrature knobs (no learnable parameters).
        self.sigrep_num_slices = config.sigrep_num_slices
        self.sigrep_t_max = config.sigrep_t_max
        self.sigrep_num_t = config.sigrep_num_t

    @override
    def _adarms_cond(
        self, time_emb: at.Float[at.Array, "b emb"], z: at.Float[at.Array, "b d"] | None
    ) -> at.Float[at.Array, "b emb"]:
        return time_emb if z is None else time_emb + self.z_proj(z)

    @override
    def get_state_representations(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions | None = None,
        timestep: at.Float[at.Array, " b"] | None = None,
        z: at.Float[at.Array, "b d"] | None = None,
    ):
        """Pi0SP has no psi head: the psi slot is served the phi rep instead.

        phi is the model's only trained state feature (action-independent,
        phi_input="state"), so downstream psi consumers (goal encoders, eval probes)
        get phi from the same prefix pass.
        """
        _, phi, kv_cache, prefix_mask, prefix_len = super().get_state_representations(
            observation, noisy_actions, timestep, z=z
        )
        return phi, phi, kv_cache, prefix_mask, prefix_len

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        batch: dict[str, Any],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """Latent self-prediction loss (MSE + SigREP sigmoid contrastive) + flow-matching action loss.

        Required batch keys: observation, actions, next_observation. phi is
        action-independent (phi_input="state", enforced by Pi0SPConfig), so no
        next_actions are needed — the action enters the forward model explicitly.
        Optional: next_is_pad (terminal masking of the prediction target).
        """
        _model.require_batch_keys(
            batch,
            ("observation", "actions", "next_observation"),
            type(self).__name__,
        )
        observation = batch["observation"]
        actions = batch["actions"]
        next_observation = batch["next_observation"]
        next_is_pad = batch.get("next_is_pad")

        # Head dropout only touches phi's current half (the target is deterministic).
        preprocess_rng, next_preprocess_rng, noise_rng, time_rng, phi_drop_rng, sigrep_rng = jax.random.split(rng, 6)
        deterministic = not train
        # Augment current/next independently (consistent with pi05crl) so phi(s,a) and the
        # bootstrap phi(s',a') cannot match on shared augmentation artifacts.
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        next_observation = _model.preprocess_observation(next_preprocess_rng, next_observation, train=train)

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

        # ---- (2) SP pass: prefix-only (phi is action-independent), stacked current+next ----
        obs_stacked = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), observation, next_observation
        )
        _, prefix_hidden, _, prefix_mask, _ = self._run_prefix(obs_stacked)

        # phi via Pi0RepBase._compute_rep from the prefix. Layer axis is 0, so slice the
        # batch on axis 1.
        def _rep_slice(sl, drop_rng, *, det):
            return self._compute_rep(
                "phi",
                prefix_hidden[:, sl],
                None,
                prefix_mask[sl],
                None,
                deterministic=det,
                dropout_rng=drop_rng,
            )

        cur = slice(None, batch_size)
        nxt = slice(batch_size, None)
        phi = _rep_slice(cur, phi_drop_rng, det=deterministic).astype(jnp.float32)
        # Actions are flattened to (b, ah*ad) to concat with the (b, d) rep.
        phi_sa = jnp.concatenate([phi, actions.reshape(batch_size, -1)], axis=-1)
        phi_pred = self.forward_proj(phi_sa)
        phi_next = jax.lax.stop_gradient(_rep_slice(nxt, None, det=True).astype(jnp.float32))

        # Mask the prediction at episode boundaries: a padded next frame is not a valid target.
        if next_is_pad is None:
            not_terminal = jnp.ones((batch_size, 1), dtype=phi_next.dtype)
        else:
            not_terminal = (1.0 - next_is_pad.astype(phi_next.dtype))[:, None]
        sp_resid = not_terminal * (phi_pred - phi_next)
        sp_loss = jnp.mean(jnp.square(sp_resid))

        # phi SigREP loss: LeJEPA's SIGReg on the (grad-carrying) current-half phi. Pushing
        # every random 1D projection of the batch toward N(0, 1) drives the embedding
        # distribution to an isotropic Gaussian — the anti-collapse complement to the MSE
        # (a constant or low-rank phi cannot be isotropic Gaussian). The stop-grad target
        # half comes from the same head, so its distribution follows without an extra term.
        sigrep_loss = _sigreg_epps_pulley(
            phi,
            sigrep_rng,
            num_slices=self.sigrep_num_slices,
            t_max=self.sigrep_t_max,
            num_t=self.sigrep_num_t,
        )

        # Diagnostics (meaned by the train loop); layer-mix entropy watches for mix collapse.
        # phi_std -> 1 and phi_mean_norm -> 0 as SIGReg pulls phi toward N(0, I).
        phi_mix_w = jax.nn.softmax(self.phi_mix.value.astype(jnp.float32))
        info = {
            "action_loss": jnp.mean(action_loss),
            "sp_loss": sp_loss,
            "sigrep_loss": sigrep_loss,
            "phi_norm": jnp.mean(jnp.linalg.norm(phi, axis=-1)),
            "phi_std": jnp.mean(jnp.std(phi, axis=0)),
            "phi_mean_norm": jnp.linalg.norm(jnp.mean(phi, axis=0)),
            "sp_resid": jnp.mean(jnp.linalg.norm(sp_resid, axis=-1)),
            "phi_mix_entropy": -jnp.sum(phi_mix_w * jnp.log(phi_mix_w + 1e-6)),
            "phi_mix_max": jnp.max(phi_mix_w),
        }
        total_loss = (
            self.action_loss_coeff * action_loss + self.sp_loss_coeff * sp_loss + self.sigrep_loss_coeff * sigrep_loss
        )
        # "rep_loss" (the combined, unweighted rep objective) keeps the train loop's shared
        # logging (which expects the CRL model's key set) working unchanged.
        return total_loss, {"rep_loss": sp_loss + sigrep_loss, **info}
