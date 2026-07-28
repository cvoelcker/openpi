"""Latent self-prediction (SP) variant of pi05.

Pi0SP trains the phi readout with a latent forward model: a stack of concat-conditioned
MLP blocks (ForwardProjHead) predicts the next-step rep from (phi(s), a). The prediction is
trained with (1) an MSE regression onto a target rep of s', expressed relative to the mean
squared target norm so that it is invariant in rep_dim, batch size and rep scale (with
normalize_sp_loss it is exactly 2 - 2*cos), and (2) sigreg, LeJEPA's SIGReg
(sketched isotropic-Gaussian regularization): random 1D projections of a small MLP
head on top of phi are pushed toward N(0, 1) via the Epps-Pulley characteristic-function
test. Following LeJEPA's own reference setup (the two-layer MLP before SIGReg; see issue
#14 on the LeJEPA repo), the isotropy constraint is applied to a projector output rather
than directly to phi — the projector absorbs the unit-variance / all-directions
constraint, letting phi concentrate variance on the directions the forward-model target
actually rewards.

CAVEAT: this indirection weakens the guarantee, and by how much is an OPEN question
upstream — LeJEPA issue #17 asks precisely this, noting that the theory wants the
*embeddings* isotropic while the implementation makes the *projections* isotropic, and
the probe then reads the embeddings. What survives the indirection is the constant-rep
collapse of the regression: a phi that is constant across the batch maps to a constant
projection, which no MLP can make isotropic. What does NOT reliably survive is pressure
against low-RANK phi: a 2-layer MLP is a dense mixing map, so by a CLT argument its 1D
projections can look Gaussian even when phi's variance is concentrated in a handful of
directions. So do not read a low `sigreg_loss` as evidence of a healthy spectrum —
`phi_eff_rank` from `batch_rep_stats` is the metric that actually covers that failure,
and `sigreg_loss_phi` below measures the size of the gap directly.

The target rep is, by default, psi(s') where psi is a frozen lagging (EMA) copy of phi — a
BYOL/JEPA-style target network that stabilizes the bootstrap (the online net would otherwise
chase its own moving output). The psi trio never receives gradients; the train loop
EMA-tracks it toward phi after every optimizer step (rep_base.update_target_networks). With
psi_lagging_ema=None there is no psi and the target falls back to the stop-gradient online
phi(s').

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
from openpi.models.rep_base import _l2_normalize
from openpi.models.rep_base import batch_rep_stats
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class BROBlock(nnx.Module):
    """One block of the latent forward model: concat the state and action embeddings, then
    a single linear + layer norm + GELU.

    Non-residual by design (an earlier revision was residual with FiLM action conditioning):
    concatenation forces every block to re-read the action rather than letting a residual
    path carry the state through untouched, so the predictor cannot degrade into an
    action-agnostic identity map. Watch `sp_loss_ratio_rand` to confirm it doesn't anyway.
    """

    def __init__(self, latent_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(2 * latent_dim, latent_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b d"], a: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        """Concatenates the action embedding to the current latent repeatedly in a FiLM like manner"""
        x = jnp.concatenate([x, a], axis=-1)
        x = self.fc1(x)
        x = self.ln1(x)
        return nnx.gelu(x)


class ForwardProjHead(nnx.Module):
    """The latent forward model: predicts phi(s') from the concatenated (phi(s), actions).

    The input is the concatenation of the state rep and the flattened action sequence; the
    output lives in the same rep_dim space as phi so it can be regressed / contrasted against
    phi(s').
    """

    def __init__(self, config: pi0_config.Pi0SPConfig, rngs: nnx.Rngs):
        blocks = config.forward_proj_blocks
        latent_dim = config.rep_dim
        action_in = config.action_dim * config.action_horizon
        self.num_blocks = blocks

        self.state_ln = nnx.LayerNorm(latent_dim, rngs=rngs)
        self.state_fc1 = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.action_fc1 = nnx.Linear(action_in, config.rep_dim, rngs=rngs)
        self.block_list = nnx.Dict({f"block_{i}": BROBlock(latent_dim, rngs) for i in range(blocks)})
        self.final = nnx.Linear(latent_dim, latent_dim, rngs=rngs)

    def _embed_state(self, x: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        """State is output of a linear layer, so we apply layer norm and GELU before the first BROBlock to give it a nonlinearity."""
        x = self.state_ln(x)
        x = nnx.gelu(x)
        return self.state_fc1(x)

    def _embed_action(self, a: at.Float[at.Array, "b ad"]) -> at.Float[at.Array, "b d"]:
        """Action only goes through a linear layer to project it into the same latent space as the state."""
        return self.action_fc1(a)

    def __call__(self, x: at.Float[at.Array, "b d"], a: at.Float[at.Array, "b ad"]) -> at.Float[at.Array, "b d"]:
        x = self._embed_state(x)
        a = self._embed_action(a)
        for i in range(self.num_blocks):
            x = self.block_list[f"block_{i}"](x, a)
        return self.final(x)


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
    _normalize_phi = False
    _normalize_psi = False

    def __init__(self, config: pi0_config.Pi0SPConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.sp_loss_coeff = config.sp_loss_coeff
        self.sigreg_loss_coeff = config.sigreg_loss_coeff
        self.z_proj = nnx.Linear(config.rep_dim, self.action_in_proj.out_features, rngs=rngs)
        self.normalize_loss = config.normalize_sp_loss
        self.target_augmentation = config.target_augmentation

        self.forward_proj = ForwardProjHead(config, rngs)

        self.sigreg_proj_fc1 = nnx.Linear(config.rep_dim, config.rep_dim, rngs=rngs)
        self.sigreg_proj_fc2 = nnx.Linear(config.rep_dim, config.rep_dim, rngs=rngs)

        # SIGReg sketch/quadrature knobs (no learnable parameters).
        self.sigreg_num_slices = config.sigreg_num_slices
        self.sigreg_t_max = config.sigreg_t_max
        self.sigreg_num_t = config.sigreg_num_t

    def _sigreg_project(self, x: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        h = self.sigreg_proj_fc1(x)
        h = nnx.gelu(h)
        return self.sigreg_proj_fc2(h)

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
        """With a lagging psi, the psi slot serves the stable EMA rep; without one, phi.

        Downstream psi consumers (goal encoders, eval probes) thus get the slowly-moving
        target rep when the target network is enabled, and the online phi otherwise —
        either way from the same single prefix pass.
        """
        psi, phi, kv_cache, prefix_mask, prefix_len = super().get_state_representations(
            observation, noisy_actions, timestep, z=z
        )
        if psi is None:
            psi = phi
        return psi, phi, kv_cache, prefix_mask, prefix_len

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        batch: dict[str, Any],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """Latent self-prediction loss (MSE onto the lagging-psi target + sigreg) + flow-matching action loss.

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
        (
            preprocess_rng,
            next_preprocess_rng,
            noise_rng,
            time_rng,
            phi_drop_rng,
            sigreg_rng,
            rand_act_rng,
        ) = jax.random.split(rng, 7)
        deterministic = not train
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, augment=self.image_augment
        )
        target_aug_rng, target_aug_train = {
            "independent": (next_preprocess_rng, train),
            "shared": (preprocess_rng, train),
            "none": (None, False),
        }[self.target_augmentation]
        next_observation = _model.preprocess_observation(
            target_aug_rng, next_observation, train=target_aug_train, augment=self.image_augment
        )

        batch_size = actions.shape[0]

        # ---- (1) Flow-matching action loss on the current observation (standard pi05) ----
        if self.action_loss_coeff > 0.0:
            noise = jax.random.normal(noise_rng, actions.shape)
            time = jax.random.beta(time_rng, 1.5, 1, (batch_size,)) * 0.999 + 0.001
            time_expanded = time[:, None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions
            suffix_out, *_ = self._forward(observation, x_t, time)
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        else:
            action_loss = jnp.zeros(actions.shape[:2], dtype=jnp.float32)

        # ---- (2) SP pass: prefix-only (phi is action-independent), stacked current+next ----
        obs_stacked = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), observation, next_observation
        )
        _, prefix_hidden, _, prefix_mask, _ = self._run_prefix(obs_stacked)

        # Reps via Pi0RepBase._compute_rep from the prefix. Layer axis is 0, so slice the
        # batch on axis 1.
        def _rep_slice(which, sl, drop_rng, *, det):
            return self._finalize_rep(
                which,
                self._compute_rep(
                    which,
                    prefix_hidden[:, sl],
                    None,
                    prefix_mask[sl],
                    None,
                    deterministic=det,
                    dropout_rng=drop_rng,
                ),
            )

        cur = slice(None, batch_size)
        nxt = slice(batch_size, None)
        phi = _rep_slice("phi", cur, phi_drop_rng, det=deterministic).astype(jnp.float32)
        # Actions are flattened to (b, ah*ad) to concat with the (b, d) rep.
        phi_pred = self.forward_proj(phi, actions.reshape(batch_size, -1))
        target_head = "psi" if self.psi_lagging_ema is not None else "phi"
        phi_next = jax.lax.stop_gradient(_rep_slice(target_head, nxt, None, det=True).astype(jnp.float32))
        phi_next_raw = phi_next

        # Mask the prediction at episode boundaries: a padded next frame is not a valid target.
        if next_is_pad is None:
            not_terminal = jnp.ones((batch_size, 1), dtype=phi_next.dtype)
        else:
            not_terminal = (1.0 - next_is_pad.astype(phi_next.dtype))[:, None]
        valid = not_terminal[:, 0]
        n_valid = jnp.maximum(jnp.sum(valid), 1.0)
        if self.normalize_loss:
            phi_pred = _l2_normalize(phi_pred)
            phi_next = _l2_normalize(phi_next)

        target_sq = jnp.sum(valid * jnp.sum(jnp.square(not_terminal * phi_next), axis=-1)) / n_valid

        def _sp_mse(pred):
            resid = not_terminal * (pred - phi_next)
            num = jnp.sum(valid * jnp.sum(jnp.square(resid), axis=-1)) / n_valid
            return num / (target_sq + 1e-8)

        sp_resid = not_terminal * (phi_pred - phi_next)
        sp_loss = _sp_mse(phi_pred)

        # ---- Action-usage diagnostic: sp_loss under uniform-noise actions ----
        rand_actions = jax.random.uniform(rand_act_rng, actions.shape, minval=-1.0, maxval=1.0)
        phi_pred_rand = jax.lax.stop_gradient(
            self.forward_proj(jax.lax.stop_gradient(phi), rand_actions.reshape(batch_size, -1))
        )
        if self.normalize_loss:
            phi_pred_rand = _l2_normalize(phi_pred_rand)
        sp_loss_rand = _sp_mse(phi_pred_rand)
        sp_loss_ratio_rand = sp_loss_rand / (sp_loss + 1e-8)

        # ---- Collapse monitor: batch alignment of the prediction (diagnostic only) ----
        pred_sg = jax.lax.stop_gradient(phi_pred)
        sq_dists = jnp.sum(jnp.square(pred_sg[:, None, :] - phi_next[None, :, :]), axis=-1)  # (b, b)
        align_logits = -sq_dists
        labels = jnp.arange(batch_size)
        align_ce = -jax.nn.log_softmax(align_logits, axis=-1)[labels, labels]
        align_acc = (jnp.argmin(sq_dists, axis=-1) == labels).astype(jnp.float32)
        # Margin: mean off-diagonal distance minus the true-pair distance (positive = aligned).
        diag_dist = sq_dists[labels, labels]
        off_diag_mean = (jnp.sum(sq_dists, axis=-1) - diag_dist) / jnp.maximum(batch_size - 1, 1)
        # Normalized rank of the true target: fraction of distractors strictly closer than it.
        # 0 = true target closest, ~0.5 = chance, 1 = worst. Unlike top-1 accuracy this is
        # comparable across batch sizes (acc mechanically drops as distractors are added).
        align_rank = jnp.sum(sq_dists < diag_dist[:, None], axis=-1).astype(jnp.float32) / jnp.maximum(
            batch_size - 1, 1
        )
        align_ce = jnp.sum(valid * align_ce) / n_valid
        align_acc = jnp.sum(valid * align_acc) / n_valid
        align_margin = jnp.sum(valid * (off_diag_mean - diag_dist)) / n_valid
        align_rank = jnp.sum(valid * align_rank) / n_valid

        phi_sigreg = self._sigreg_project(phi)
        sigreg_loss = _sigreg_epps_pulley(
            phi_sigreg,
            sigreg_rng,
            num_slices=self.sigreg_num_slices,
            t_max=self.sigreg_t_max,
            num_t=self.sigreg_num_t,
        )

        # Diagnostic only (stop_gradient, never added to the loss): the same SIGReg statistic
        # evaluated directly on phi. LeJEPA issue #17's open question is whether constraining
        # the projector actually constrains the embedding; this is that gap, measured. Read it
        # against `sigreg_loss`:
        #   both small          -> isotropy is reaching phi; the projector is not laundering.
        #   sigreg_loss ~ 0 but
        #   sigreg_loss_phi big -> the MLP is absorbing the constraint and phi is unconstrained.
        #                          Cross-check phi_eff_rank; if that is also low, the isotropy
        #                          term is buying nothing and the anti-collapse pressure has to
        #                          come from somewhere else.
        # Reuses sigreg_rng so both statistics see the SAME random slice directions, making the
        # two numbers directly comparable rather than differing by sketch noise.
        sigreg_loss_phi = _sigreg_epps_pulley(
            jax.lax.stop_gradient(phi),
            sigreg_rng,
            num_slices=self.sigreg_num_slices,
            t_max=self.sigreg_t_max,
            num_t=self.sigreg_num_t,
        )

        phi_mix_w = jax.nn.softmax(self.phi_mix.value.astype(jnp.float32))

        # Self similarity: how close phi(s) already is to its own target phi(s')
        self_sim = jnp.sum(valid * jnp.sum(_l2_normalize(phi) * _l2_normalize(phi_next_raw), axis=-1)) / n_valid

        info = {
            "action_loss": jnp.mean(action_loss),
            "sp_loss": sp_loss,
            "sigreg_loss": sigreg_loss,
            "sigreg_loss_phi": sigreg_loss_phi,
            "sigreg_loss_scaled": batch_size * sigreg_loss,
            "sp_loss_weighted": self.sp_loss_coeff * sp_loss,
            "sigreg_loss_weighted": self.sigreg_loss_coeff * sigreg_loss,
            "phi_norm": jnp.mean(jnp.linalg.norm(phi, axis=-1)),
            "target_norm": jnp.mean(jnp.linalg.norm(phi_next_raw, axis=-1)),
            **batch_rep_stats(phi, "phi"),
            **batch_rep_stats(phi_next_raw, "target"),
            **batch_rep_stats(phi_sigreg, "sigreg"),
            "sp_resid": jnp.mean(jnp.linalg.norm(sp_resid, axis=-1)),
            "sp_loss_rand_actions": sp_loss_rand,
            "sp_loss_ratio_rand": sp_loss_ratio_rand,
            "sp_align_ce": align_ce,
            "sp_align_acc": align_acc,
            "sp_align_margin": align_margin,
            "sp_align_rank": align_rank,
            "phi_mix_entropy": -jnp.sum(phi_mix_w * jnp.log(phi_mix_w + 1e-6)),
            "phi_mix_max": jnp.max(phi_mix_w),
            "phi_self_sim": self_sim,
        }
        total_loss = (
            self.action_loss_coeff * action_loss + self.sp_loss_coeff * sp_loss + self.sigreg_loss_coeff * sigreg_loss
        )
        return total_loss, {"rep_loss": sp_loss + sigreg_loss, **info}
