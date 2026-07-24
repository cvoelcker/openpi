"""Latent self-prediction (SP) variant of pi05.

Pi0SP trains the phi readout with a latent forward model: a BRO-style residual MLP
(ForwardProjHead) predicts the next-step rep from (phi(s), a). The prediction is trained
with (1) an MSE regression onto a target rep of s' and (2) sigreg, LeJEPA's SIGReg
(sketched isotropic-Gaussian regularization): random 1D projections of a small MLP
head on top of phi are pushed toward N(0, 1) via the Epps-Pulley characteristic-function
test. Following LeJEPA (and issue #17 on the LeJEPA repo), the isotropy constraint is
applied to a projector output rather than directly to phi — the projector absorbs the
unit-variance / all-directions constraint, letting phi concentrate variance on the
directions the forward-model target actually rewards. Isotropy at the projector still
rules out the constant-rep collapse of the regression (a collapsed phi cannot yield an
isotropic projection either).

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
from openpi.models.rep_base import batch_rep_stats
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class BROBlock(nnx.Module):
    """A BRO-style residual block with a linear layer, layer norm, and GELU activation.
    The input is added to the output to form the final output. This is used in the ForwardProjHead
    to ease learning the identity function when the action is not informative.

    WARNING: temporally changed to non reidual for experiment
    """

    def __init__(self, latent_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.ln1 = nnx.LayerNorm(latent_dim, rngs=rngs)
        self.fc2 = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(latent_dim, rngs=rngs)

        self.film_beta = nnx.Linear(latent_dim, latent_dim, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b d"], a: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        residual = x
        x = self.fc1(x)
        x = self.ln1(x)
        x = nnx.gelu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        beta = self.film_beta(a)
        x = x + beta
        return x + residual


class ForwardProjHead(nnx.Module):
    """The latent forward model: predicts phi(s') from the concatenated (phi(s), actions).

    The input is the concatenation of the state rep and the flattened action sequence; the
    output lives in the same rep_dim space as phi so it can be regressed / contrasted against
    phi(s').
    """

    def __init__(self, config: pi0_config.Pi0SPConfig, rngs: nnx.Rngs):
        blocks = config.forward_proj_blocks
        latent_dim = config.rep_dim
        self.state_fc1 = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        self.state_ln1 = nnx.LayerNorm(latent_dim, rngs=rngs)
        self.state_fc2 = nnx.Linear(latent_dim, latent_dim, rngs=rngs)
        action_in = config.action_dim * config.action_horizon
        self.action_fc1 = nnx.Linear(action_in, config.rep_dim, rngs=rngs)
        self.action_ln1 = nnx.LayerNorm(config.rep_dim, rngs=rngs)
        self.action_fc2 = nnx.Linear(config.rep_dim, config.rep_dim, rngs=rngs)
        self.action_ln2 = nnx.LayerNorm(config.rep_dim, rngs=rngs)
        self.num_blocks = blocks
        self.block_list = nnx.Dict({f"block_{i}": BROBlock(latent_dim, rngs) for i in range(blocks)})

    def _embed_state(self, x: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        x = self.state_fc1(x)
        x = self.state_ln1(x)
        x = nnx.gelu(x)
        return self.state_fc2(x)

    def _embed_action(self, a: at.Float[at.Array, "b ad"]) -> at.Float[at.Array, "b d"]:
        a = self.action_fc1(a)
        a = self.action_ln1(a)
        a = nnx.gelu(a)
        a = self.action_fc2(a)
        return self.action_ln2(a)

    def __call__(self, x: at.Float[at.Array, "b d"], a: at.Float[at.Array, "b ad"]) -> at.Float[at.Array, "b d"]:
        x = self._embed_state(x)
        a = self._embed_action(a)
        for i in range(self.num_blocks):
            x = self.block_list[f"block_{i}"](x, a)
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
    # phi stays unnormalized: SIGReg now acts on a projector output rather than phi
    # directly, but phi's scale is still pinned by the MSE regression onto the (lagging)
    # target head, and L2-normalizing would still fight that regression. psi (the lagging
    # EMA copy of phi, when built) mirrors phi's policy: the regression target must live
    # in the same raw space.
    _normalize_phi = False
    _normalize_psi = False

    def __init__(self, config: pi0_config.Pi0SPConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.sp_loss_coeff = config.sp_loss_coeff
        self.sigreg_loss_coeff = config.sigreg_loss_coeff
        # z conditions the action expert via the AdaRMS pathway (keeps the prefix z-free).
        # Unused by compute_loss (no z is passed); kept for inference-time steering.
        self.z_proj = nnx.Linear(config.rep_dim, self.action_in_proj.out_features, rngs=rngs)

        self.forward_proj = ForwardProjHead(config, rngs)

        # 2-layer projection head that absorbs the SIGReg isotropy constraint (issue #17
        # on the LeJEPA repo). Same output dim as phi so it acts as a JEPA-style
        # predictor head rather than a low-dim bottleneck. Only gets gradients from the
        # SIGReg term — the forward-model / MSE path never sees it.
        self.sigreg_proj_fc1 = nnx.Linear(config.rep_dim, config.rep_dim, rngs=rngs)
        self.sigreg_proj_ln = nnx.LayerNorm(config.rep_dim, rngs=rngs)
        self.sigreg_proj_fc2 = nnx.Linear(config.rep_dim, config.rep_dim, rngs=rngs)

        # SIGReg sketch/quadrature knobs (no learnable parameters).
        self.sigreg_num_slices = config.sigreg_num_slices
        self.sigreg_t_max = config.sigreg_t_max
        self.sigreg_num_t = config.sigreg_num_t

    def _sigreg_project(self, x: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b d"]:
        h = self.sigreg_proj_fc1(x)
        h = self.sigreg_proj_ln(h)
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

        # Reps via Pi0RepBase._compute_rep from the prefix. Layer axis is 0, so slice the
        # batch on axis 1.
        def _rep_slice(which, sl, drop_rng, *, det):
            return self._compute_rep(
                which,
                prefix_hidden[:, sl],
                None,
                prefix_mask[sl],
                None,
                deterministic=det,
                dropout_rng=drop_rng,
            )

        cur = slice(None, batch_size)
        nxt = slice(batch_size, None)
        phi = _rep_slice("phi", cur, phi_drop_rng, det=deterministic).astype(jnp.float32)
        # Actions are flattened to (b, ah*ad) to concat with the (b, d) rep.
        phi_pred = self.forward_proj(phi, actions.reshape(batch_size, -1))
        # Prediction target: the lagging EMA psi(s') when the target network is enabled
        # (psi gets no gradients anyway, but stop_gradient also severs the backbone path
        # through the next-half hiddens), else the stop-gradient online phi(s').
        target_head = "psi" if self.psi_lagging_ema is not None else "phi"
        phi_next = jax.lax.stop_gradient(_rep_slice(target_head, nxt, None, det=True).astype(jnp.float32))

        # Mask the prediction at episode boundaries: a padded next frame is not a valid target.
        if next_is_pad is None:
            not_terminal = jnp.ones((batch_size, 1), dtype=phi_next.dtype)
        else:
            not_terminal = (1.0 - next_is_pad.astype(phi_next.dtype))[:, None]
        sp_resid = not_terminal * (phi_pred - phi_next)
        sp_loss = jnp.mean(jnp.square(sp_resid))

        # ---- Collapse monitor: batch alignment of the prediction (diagnostic only) ----
        # Is phi_pred(s,a) meaningfully closer to its own target phi_next than to the other
        # targets in the batch? Under collapse all targets look alike, so top-1 accuracy
        # falls to chance (1/b) and the InfoNCE-style cross-entropy rises to log(b). Fully
        # stop-gradient: it never contributes to the training loss.
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
        # Rows with a padded next frame have no valid target; exclude them from the averages.
        valid = not_terminal[:, 0]
        n_valid = jnp.maximum(jnp.sum(valid), 1.0)
        align_ce = jnp.sum(valid * align_ce) / n_valid
        align_acc = jnp.sum(valid * align_acc) / n_valid
        align_margin = jnp.sum(valid * (off_diag_mean - diag_dist)) / n_valid
        align_rank = jnp.sum(valid * align_rank) / n_valid

        # ---- Action-usage diagnostic: sp_loss under uniform-noise actions ----
        # If the forward model ignores `a`, sp_loss with random actions matches sp_loss with
        # the real ones and the ratio sits near 1 — meaning action-conditioning is nominal
        # and the predictor is effectively self-predictive. A ratio well above 1 means the
        # actions carry real information for the target. Fully stop-gradient; diagnostic only.
        rand_actions = jax.random.uniform(rand_act_rng, actions.shape, minval=-1.0, maxval=1.0)
        phi_pred_rand = jax.lax.stop_gradient(self.forward_proj(jax.lax.stop_gradient(phi), rand_actions.reshape(batch_size, -1)))
        sp_resid_rand = not_terminal * (phi_pred_rand - phi_next)
        sp_loss_rand = jnp.mean(jnp.square(sp_resid_rand))
        sp_loss_ratio_rand = sp_loss_rand / (sp_loss + 1e-8)

        # phi sigreg loss: LeJEPA's SIGReg on a 2-layer projection of the (grad-carrying)
        # current-half phi (issue #17 fix — the projector absorbs the unit-variance /
        # all-directions constraint so phi is free to concentrate variance on the
        # forward-model-relevant directions). Isotropy at the projector still forbids a
        # constant / low-rank phi (a collapsed phi cannot yield an isotropic projection).
        phi_sigreg = self._sigreg_project(phi)
        sigreg_loss = _sigreg_epps_pulley(
            phi_sigreg,
            sigreg_rng,
            num_slices=self.sigreg_num_slices,
            t_max=self.sigreg_t_max,
            num_t=self.sigreg_num_t,
        )

        # Diagnostics (meaned by the train loop); layer-mix entropy watches for mix collapse.
        # sigreg_std -> 1 and sigreg_mean_norm -> 0 as SIGReg pulls the projector output
        # (not phi itself, post issue-#17 fix) toward N(0, I).
        phi_mix_w = jax.nn.softmax(self.phi_mix.value.astype(jnp.float32))
        info = {
            "action_loss": jnp.mean(action_loss),
            "sp_loss": sp_loss,
            "sigreg_loss": sigreg_loss,
            # Batch-size-independent Epps-Pulley statistic (classical n-scaling). Under
            # perfect Gaussianity this converges to the estimator's sampling-noise floor
            # of 1 - 1/sqrt(3) ~= 0.42 (for large t_max) regardless of batch size; values
            # well above that indicate genuine non-Gaussianity.
            "sigreg_loss_scaled": batch_size * sigreg_loss,
            "phi_norm": jnp.mean(jnp.linalg.norm(phi, axis=-1)),
            "target_norm": jnp.mean(jnp.linalg.norm(phi_next, axis=-1)),
            **batch_rep_stats(phi, "phi"),
            **batch_rep_stats(phi_next, "target"),
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
        }
        total_loss = (
            self.action_loss_coeff * action_loss + self.sp_loss_coeff * sp_loss + self.sigreg_loss_coeff * sigreg_loss
        )
        # "rep_loss" (the combined, unweighted rep objective) keeps the train loop's shared
        # logging (which expects the CRL model's key set) working unchanged.
        return total_loss, {"rep_loss": sp_loss + sigreg_loss, **info}
