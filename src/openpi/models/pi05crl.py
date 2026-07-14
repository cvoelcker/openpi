"""Contrastive-RL (CRL) variant of pi05.

Pi0CRL adds a symmetric InfoNCE loss between the phi (current-time anchor) and psi
(future-time target) readouts of Pi0RepBase, with a learnable CLIP-style temperature,
alongside the standard flow-matching action loss.
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
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# Batch keys the val-time ranking probe needs (produced by rl_data_loader).
_RANKING_KEYS = (
    "frame_index",
    "next_frame_index",
    "next_is_pad",
    "future_frame_index",
    "future_is_pad",
    "future_episode_id",
    "goal_frame_index",
    "goal_is_pad",
    "negative_frame_index",
    "negative_episode_id",
)


class Pi0CRL(Pi0RepBase):
    def __init__(self, config: pi0_config.Pi0CRLConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.crl_loss_coeff = config.crl_loss_coeff

        # Learnable CRL temperature. Reps are L2-normalized before the InfoNCE dot products,
        # so logits are cosine sims; logit_scale = log(1/temperature) rescales them (CLIP-style),
        # clamped at exp(logit_scale) <= 100 to avoid runaway. Named to match the freeze filter.
        self.logit_scale = nnx.Param(jnp.asarray(jnp.log(1.0 / config.crl_temperature_init), dtype=jnp.float32))
        self.logsumexp_penalty_coeff = config.logsumexp_penalty_coeff

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        batch: dict[str, Any],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """CRL InfoNCE + flow-matching action loss.

        Required batch keys: observation, future_observation, actions.
        Optional: episode_id, negative_observation; at val time (train=False) additionally
        next_observation / goal_observation and the frame-index keys enable the ranking probe.
        """
        _model.require_batch_keys(batch, ("observation", "future_observation", "actions"), type(self).__name__)
        observation = batch["observation"]
        future_observation = batch["future_observation"]
        actions = batch["actions"]
        episode_id = batch.get("episode_id")
        negative_observation = batch.get("negative_observation")

        # Val ranking probe: score the batch's own next/goal/negative draws against the anchor
        # and check temporal ordering. Needs the frame indices from the data loader; train
        # batches and older cached loaders without them just skip the probe.
        do_ranking = not train and episode_id is not None and "frame_index" in batch and "future_episode_id" in batch
        ranking_indices = {k: batch[k] for k in _RANKING_KEYS if k in batch} if do_ranking else None
        next_observation = batch.get("next_observation") if do_ranking else None
        goal_observation = batch.get("goal_observation") if do_ranking else None

        preprocess_rng, future_preprocess_rng, aux_preprocess_rng, noise_rng, time_rng, dropout_rng = jax.random.split(
            rng, 6
        )
        negative_preprocess_rng, next_preprocess_rng, goal_preprocess_rng = jax.random.split(aux_preprocess_rng, 3)
        # Independent head-dropout keys per rep (only used when train=True).
        phi_drop_rng, psi_drop_rng, neg_drop_rng, rank_drop_rng = jax.random.split(dropout_rng, 4)
        deterministic = not train
        # Augment each contrastive view independently — a shared key would apply the
        # identical crop/rotation/jitter to each anchor/future pair, letting the model match
        # on shared appearance artifacts instead of learning temporal state progress.
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        future_observation = _model.preprocess_observation(future_preprocess_rng, future_observation, train=train)

        # Stack current + future (+ optional within-task negative, + optional next/goal for the
        # val ranking probe) for one forward pass. `slots` records each view's position in the
        # stack. Actions are duplicated for the non-current views — those suffixes are discarded,
        # so the duplicated actions never enter the action loss.
        obs_group = [observation, future_observation]
        slots = {"future": 1}
        for name, obs, obs_rng in (
            ("negative", negative_observation, negative_preprocess_rng),
            ("next", next_observation, next_preprocess_rng),
            ("goal", goal_observation, goal_preprocess_rng),
        ):
            if obs is not None:
                slots[name] = len(obs_group)
                obs_group.append(_model.preprocess_observation(obs_rng, obs, train=train))
        observation_stacked = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *obs_group)
        actions_stacked = jnp.concatenate([actions] * len(obs_group), axis=0)

        orig_batch_size = actions.shape[0]
        noise = jax.random.normal(noise_rng, actions_stacked.shape)
        time = jax.random.beta(time_rng, 1.5, 1, (actions_stacked.shape[0],)) * 0.999 + 0.001
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions_stacked
        u_t = noise - actions_stacked

        suffix_out, prefix_hidden, suffix_hidden, prefix_mask, suffix_mask = self._forward(
            observation_stacked, x_t, time
        )

        # Action loss: current-obs half only.
        v_t = self.action_out_proj(suffix_out[:orig_batch_size, -self.action_horizon :])
        action_loss = jnp.mean(jnp.square(v_t - u_t[:orig_batch_size]), axis=-1)

        # CRL reps via dedicated heads over a learned layer-mix (see Pi0RepBase._compute_rep;
        # each rep reads the prefix or suffix per its configured input source). Layer axis is 0,
        # so slice batch on axis 1. The stack is [current | future | ...slots], so view k
        # occupies [k*B:(k+1)*B].
        B = orig_batch_size

        def _rep_slot(which, k, drop_rng):
            return self._compute_rep(
                which,
                prefix_hidden[:, k * B : (k + 1) * B],
                suffix_hidden[:, k * B : (k + 1) * B],
                prefix_mask[k * B : (k + 1) * B],
                suffix_mask[k * B : (k + 1) * B],
                deterministic=deterministic,
                dropout_rng=drop_rng,
            )  # (B, d)

        def _psi_slot(name, drop_rng):
            return _rep_slot("psi", slots[name], drop_rng)

        psi_raw = _psi_slot("future", psi_drop_rng)
        phi_raw = _rep_slot("phi", 0, phi_drop_rng)

        # L2-normalize reps onto the unit sphere so InfoNCE logits are cosine similarities.
        # This removes the norm-inflation shortcut (driving train loss down by growing ||rep||
        # rather than improving alignment) and forces the heads to separate directions. Cast to
        # float32 first — reps come out of the projection in the (bf16) embed dtype, and the
        # normalization + logit/logsumexp math wants full precision.
        phi_raw = phi_raw.astype(jnp.float32)
        psi_raw = psi_raw.astype(jnp.float32)
        phi_flat = _l2_normalize(phi_raw)
        psi_flat = _l2_normalize(psi_raw)
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
            psi_neg = _l2_normalize(_psi_slot("negative", neg_drop_rng))
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
            "phi_mix_entropy": -jnp.sum(phi_mix_w * jnp.log(phi_mix_w + 1e-6)),
            "psi_mix_entropy": -jnp.sum(psi_mix_w * jnp.log(psi_mix_w + 1e-6)),
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

        if ranking_indices is not None and episode_id is not None:
            # Val ranking probe (compute-only, NOT in total_loss). The batch's own draws give up
            # to 4 same-episode future candidates at KNOWN offsets (next, future, goal, and the
            # negative when it lands in the anchor's future) plus non-future candidates (negative
            # landing in the past / another episode). Two capabilities, measured directly:
            #   (a) rank_order_*: temporally NEARER same-episode futures must outscore farther
            #       ones (occupancy decays geometrically with offset). Scene fingerprints are
            #       constant within an episode, so this is shortcut-proof progress measurement.
            #   (b) rank_neg_*: any same-episode future must outscore zero-occupancy candidates
            #       (same-episode past/current frames, other-episode frames).
            # Unlike raw val InfoNCE, none of this depends on batch composition or collisions.
            anchor_idx = ranking_indices["frame_index"].astype(jnp.int32)
            next_drop_rng, goal_drop_rng = jax.random.split(rank_drop_rng)

            def _rank_cos(name, drop_rng):
                r = _l2_normalize(_psi_slot(name, drop_rng))
                return jnp.sum(phi_flat * r, axis=-1)  # (B,) cosine score against the anchor

            # Candidate lists: score, temporal offset (frames), and validity as a future candidate.
            cand_scores = [jnp.sum(phi_flat * psi_flat, axis=-1)]
            cand_offsets = [ranking_indices["future_frame_index"].astype(jnp.int32) - anchor_idx]
            # future_episode_id differs from episode_id only in random_future_control mode.
            cand_valid = [~ranking_indices["future_is_pad"] & (ranking_indices["future_episode_id"] == episode_id)]
            if "next" in slots:
                cand_scores.append(_rank_cos("next", next_drop_rng))
                cand_offsets.append(ranking_indices["next_frame_index"].astype(jnp.int32) - anchor_idx)
                cand_valid.append(~ranking_indices["next_is_pad"])
            if "goal" in slots:
                cand_scores.append(_rank_cos("goal", goal_drop_rng))
                cand_offsets.append(ranking_indices["goal_frame_index"].astype(jnp.int32) - anchor_idx)
                cand_valid.append(~ranking_indices["goal_is_pad"])
            neg_same_ep = neg_nonfuture_past = None
            if crl_self_neg is not None:
                d_neg = ranking_indices["negative_frame_index"].astype(jnp.int32) - anchor_idx
                neg_same_ep = ranking_indices["negative_episode_id"] == episode_id
                s_neg_cos = jnp.sum(phi_flat * psi_neg, axis=-1)
                # A same-episode strict-future negative joins the ordering candidates (last row).
                cand_scores.append(s_neg_cos)
                cand_offsets.append(d_neg)
                cand_valid.append(neg_same_ep & (d_neg > 0))
                neg_nonfuture_past = neg_same_ep & (d_neg <= 0)

            S = jnp.stack(cand_scores)  # (K, B)
            D = jnp.stack(cand_offsets)  # (K, B)
            V = jnp.stack(cand_valid)  # (K, B)

            def _safe_mean(x, mask):
                mask = mask.astype(jnp.float32)
                return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1.0)

            # (a) ordering accuracy over candidate pairs, each counted once (i < j).
            once = jnp.tril(jnp.ones((S.shape[0], S.shape[0]), dtype=bool), k=-1)[..., None]
            margin = jnp.where(D[:, None] < D[None, :], S[:, None] - S[None, :], S[None, :] - S[:, None])
            pair_valid = V[:, None] & V[None, :] & (D[:, None] != D[None, :]) & once
            diagnostics["rank_order_acc"] = _safe_mean((margin > 0).astype(jnp.float32), pair_valid)
            diagnostics["rank_order_margin"] = _safe_mean(margin, pair_valid)
            diagnostics["rank_order_pairs"] = jnp.sum(pair_valid) / B
            rank_losses = [(jax.nn.softplus(-logit_scale * margin), pair_valid)]

            # (b) future candidates vs the negative when it is a zero-occupancy (non-future) frame.
            if crl_self_neg is not None:
                fut_vs_neg = S[:-1] - s_neg_cos[None, :]  # (K-1, B); last row is the neg itself
                for tag, cond in (("rank_neg_other", ~neg_same_ep), ("rank_neg_past", neg_nonfuture_past)):
                    valid = V[:-1] & cond[None, :]
                    diagnostics[f"{tag}_acc"] = _safe_mean((fut_vs_neg > 0).astype(jnp.float32), valid)
                    diagnostics[f"{tag}_pairs"] = jnp.sum(valid) / B
                    rank_losses.append((jax.nn.softplus(-logit_scale * fut_vs_neg), valid))

            # Aggregate pairwise logistic ranking loss (same logit scale as the InfoNCE, so the
            # units are comparable to rep_loss; 0 = every comparison correctly ordered w/ margin).
            total_l = sum(jnp.sum(loss * v.astype(jnp.float32)) for loss, v in rank_losses)
            total_n = sum(jnp.sum(v) for _, v in rank_losses)
            diagnostics["ranking_loss"] = total_l / jnp.maximum(total_n.astype(jnp.float32), 1.0)

        total_loss = self.action_loss_coeff * action_loss + self.crl_loss_coeff * crl_loss
        return total_loss, {"action_loss": jnp.mean(action_loss), "rep_loss": crl_loss, **diagnostics}
