"""Self-prediction (SP) variant of pi05 with a TD-learned value head.

Pi0SPTD is Pi0SP plus a small head on phi that predicts the probability that the current
episode ends in success. The head is a two-layer MLP shaped exactly like the SIGReg
projector, but its output is a two-bin categorical trained with cross-entropy: HL-Gauss with
the support degenerated to {0, 1}, so the two-hot target for a scalar TD target y in [0, 1]
is simply [1 - y, y]. The value is read back as the bin-weighted expectation, i.e. the
softmax probability of bin 1. Keeping the categorical form (rather than a scalar sigmoid)
makes an extension to N bins mechanical. With value_normalize_input (the default) the head
sees the L2-normalized rep, matching how downstream consumers query phi.

Reward placement: an anchor is treated as DONE when its action chunk reaches the end of the
episode — `next_is_pad`, the last ~action_horizon anchors — not just on the final frame.
Those rows take the episode label directly; every other row has zero reward and bootstraps
gamma * V(s'). Bootstrapping the done rows instead would spend a full gamma on a gap of
fewer than action_horizon steps, so the discount would not mean the same thing there as
elsewhere. This way every surviving bootstrap spans exactly one chunk.

Failure is a timeout, not a terminal: an unsuccessful episode was truncated at the step
limit, so its last chunk carries no reward and has no successor to bootstrap from. Those
rows are dropped from the loss (value_truncated_failure) instead of being regressed to 0.

With value_action_conditioned the head learns Q(s, a) by SARSA instead of V(s) by TD: the
flattened action chunk is a second input to the value head (widening its first layer only),
and the bootstrap scores s' under the dataset's own a' (`next_actions`).

The bootstrap reads the rep Pi0SP already computes as its self-prediction target (the
lagging psi EMA rep with psi_lagging_ema set, the stop-gradient online phi otherwise), so
the value head adds no backbone pass. The head itself is online; if TD proves unstable, the
upgrade is an EMA copy of the head along the lines of Pi0SF's _SFTarget machinery.
"""

import logging
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi05self_pred import Pi0SP
from openpi.models.rep_base import _l2_normalize
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def _masked_mean(x: at.Array, mask: at.Array) -> at.Array:
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1.0)


class Pi0SPTD(Pi0SP):
    def __init__(self, config: pi0_config.Pi0SPTDConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.value_loss_coeff = config.value_loss_coeff
        self.value_gamma = config.value_gamma
        self.value_stop_grad_phi = config.value_stop_grad_phi
        self.value_normalize_input = config.value_normalize_input
        self.value_action_conditioned = config.value_action_conditioned
        self.value_truncated_failure = config.value_truncated_failure

        # Two logits over the value bins {0, 1}. Shaped after _sigreg_project. In Q mode the
        # flattened action chunk is a second input to the head, so the first layer widens by
        # the action size; nothing else in the model is touched.
        action_in = config.action_dim * config.action_horizon if config.value_action_conditioned else 0
        self.value_head_fc1 = nnx.Linear(config.rep_dim + action_in, config.rep_dim, rngs=rngs)
        self.value_head_fc2 = nnx.Linear(config.rep_dim, 2, rngs=rngs)

    def _value_logits(
        self,
        x: at.Float[at.Array, "b d"],
        actions: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> at.Float[at.Array, "b 2"]:
        x = x.astype(jnp.float32)
        if self.value_normalize_input:
            # Pi0SP keeps phi unnormalized; downstream consumers score the unit rep, so the head
            # is trained on the same input it will be queried with.
            x = _l2_normalize(x)
        if self.value_action_conditioned:
            # Actions arrive already normalized by the data pipeline, on the same scale as the
            # flow-matching target, so they enter the head as-is.
            x = jnp.concatenate([x, actions.reshape(actions.shape[0], -1).astype(jnp.float32)], axis=-1)
        h = self.value_head_fc1(x.astype(jnp.float32))
        h = nnx.gelu(h)
        return self.value_head_fc2(h).astype(jnp.float32)

    @staticmethod
    def _value(logits: at.Float[at.Array, "b 2"]) -> at.Float[at.Array, " b"]:
        """Bin-weighted expectation over the {0, 1} support = P(bin 1)."""
        return jax.nn.softmax(logits, axis=-1)[..., 1]

    @staticmethod
    def _two_hot_ce(logits: at.Float[at.Array, "b 2"], y: at.Float[at.Array, " b"]) -> at.Float[at.Array, " b"]:
        """Cross-entropy against the two-hot encoding [1 - y, y] of a target y in [0, 1]."""
        log_p = jax.nn.log_softmax(logits, axis=-1)
        return -((1.0 - y) * log_p[..., 0] + y * log_p[..., 1])

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        batch: dict[str, Any],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """Pi0SP's losses plus a TD cross-entropy loss on the value head.

        Required batch keys: Pi0SP's (observation, actions, next_observation) plus
        `episode_success` and `next_is_pad`, and `next_actions` when value_action_conditioned.
        """
        _model.require_batch_keys(batch, ("episode_success", "next_is_pad"), type(self).__name__)
        if self.value_action_conditioned:
            _model.require_batch_keys(batch, ("next_actions",), type(self).__name__)

        total_loss, info, aux = self._sp_terms(rng, batch, train=train)
        phi, phi_next = aux["phi"], aux["phi_next"]

        success = batch["episode_success"].astype(jnp.float32)
        # `done` is next_is_pad, not is_terminal: an anchor whose action chunk reaches the end of
        # the episode is treated as terminal and takes the label directly. Bootstrapping those
        # rows would apply one full gamma across a gap of fewer than action_horizon steps, so the
        # discount would not mean the same thing on them as everywhere else. This keeps every
        # surviving bootstrap exactly one chunk wide.
        done = batch["next_is_pad"].astype(jnp.float32)

        # SARSA in Q mode: the bootstrap scores s' under the dataset's own a'. phi_next is
        # already stop-gradded by Pi0SP, so the target stays semi-gradient either way.
        next_actions = batch["next_actions"] if self.value_action_conditioned else None
        v_next = jax.lax.stop_gradient(self._value(self._value_logits(phi_next, next_actions)))
        target = done * success + (1.0 - done) * self.value_gamma * v_next

        head_input = jax.lax.stop_gradient(phi) if self.value_stop_grad_phi else phi
        logits = self._value_logits(head_input, batch["actions"] if self.value_action_conditioned else None)
        # A failed episode ends on a timeout, not on a state that is definitively worthless:
        # the run was truncated at the step limit. There is no reward to regress there and no
        # successor to bootstrap from, so the row is dropped rather than pinned to 0. Successful
        # terminals keep their label. Every other row contributes.
        loss_mask = jnp.ones_like(done)
        if self.value_truncated_failure:
            loss_mask = 1.0 - done * (1.0 - success)
        value_loss = _masked_mean(self._two_hot_ce(logits, target), loss_mask)

        value = self._value(logits)
        value_info = {
            "value_loss": value_loss,
            "value_mean": jnp.mean(value),
            # The separation between these two is the metric that says whether the head learned
            # anything beyond the marginal success rate.
            "value_mean_success": _masked_mean(value, success),
            "value_mean_failure": _masked_mean(value, 1.0 - success),
            "value_td_resid": jnp.mean(jnp.abs(value - target)),
            # Fraction of rows taking the label rather than bootstrapping (~action_horizon
            # anchors per episode), and the value the head assigns them.
            "value_terminal_frac": jnp.mean(done),
            "value_at_terminal": _masked_mean(value, done),
            # Fraction of rows the loss actually sees; drops below 1 by the truncated failures.
            "value_loss_frac": jnp.mean(loss_mask),
        }
        if self.value_action_conditioned:
            # Q under uniform-noise actions, mirroring Pi0SP's sp_loss_rand diagnostic: if the
            # gap is ~0 the head collapsed to V(s) and the action input is being ignored. Keyed
            # off rng without splitting it, so the SP terms' random stream is untouched.
            rand_actions = jax.random.uniform(
                jax.random.fold_in(rng, 0x9), batch["actions"].shape, minval=-1.0, maxval=1.0
            )
            q_rand = self._value(self._value_logits(jax.lax.stop_gradient(phi), rand_actions))
            value_info["value_action_gap"] = jnp.mean(value - q_rand)
        total_loss = total_loss + self.value_loss_coeff * value_loss
        return total_loss, {**info, **value_info, "rep_loss": info["rep_loss"] + value_loss}
