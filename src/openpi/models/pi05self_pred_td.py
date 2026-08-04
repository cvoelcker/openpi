"""Self-prediction (SP) variant of pi05 with a TD-learned value head.

Pi0SPTD is Pi0SP plus a small head on phi that predicts the probability that the current
episode ends in success. The head is a two-layer MLP shaped exactly like the SIGReg
projector, but its output is a two-bin categorical trained with cross-entropy: HL-Gauss with
the support degenerated to {0, 1}, so the two-hot target for a scalar TD target y in [0, 1]
is simply [1 - y, y]. The value is read back as the bin-weighted expectation, i.e. the
softmax probability of bin 1. Keeping the categorical form (rather than a scalar sigmoid)
makes an extension to N bins mechanical.

Reward placement: the 0/1 label sits on the TERMINAL STATE — the episode's last frame, one
step per episode. Every other anchor has zero reward and bootstraps gamma * V(s'). Note that
`next_is_pad` is NOT the terminal flag: it marks the last ~action_horizon anchors, whose
next frame was clamped to the episode's last one. Those anchors bootstrap off exactly the
frame that carries the label, which is how the signal propagates backwards.

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
        self.value_terminal_aux = config.value_terminal_aux

        # Two logits over the value bins {0, 1}. Shaped after _sigreg_project.
        self.value_head_fc1 = nnx.Linear(config.rep_dim, config.rep_dim, rngs=rngs)
        self.value_head_fc2 = nnx.Linear(config.rep_dim, 2, rngs=rngs)

    def _value_logits(self, x: at.Float[at.Array, "b d"]) -> at.Float[at.Array, "b 2"]:
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
        `episode_success` and `is_terminal`. Optional: next_is_pad.
        """
        _model.require_batch_keys(batch, ("episode_success", "is_terminal"), type(self).__name__)

        total_loss, info, aux = self._sp_terms(rng, batch, train=train)
        phi, phi_next = aux["phi"], aux["phi_next"]

        success = batch["episode_success"].astype(jnp.float32)
        # `done` is is_terminal, NOT next_is_pad and NOT Pi0SP's not_terminal mask: the reward
        # exists only on the episode's last frame. Terminal anchors must also be kept out of the
        # bootstrap for a second reason -- RandomFutureDataset returns next_idx == t there, so
        # their "bootstrap" would be self-referential.
        done = batch["is_terminal"].astype(jnp.float32)

        v_next = jax.lax.stop_gradient(self._value(self._value_logits(phi_next)))
        target = done * success + (1.0 - done) * self.value_gamma * v_next

        value_input = jax.lax.stop_gradient(phi) if self.value_stop_grad_phi else phi
        logits = self._value_logits(value_input)
        # Every row contributes -- unlike the self-prediction loss, nothing is masked out.
        value_loss = jnp.mean(self._two_hot_ce(logits, target))

        if self.value_terminal_aux:
            # phi_next is already stop-gradded, so this only trains the value head.
            next_is_terminal = (
                jnp.zeros_like(done) if (pad := batch.get("next_is_pad")) is None else pad.astype(jnp.float32)
            )
            aux_ce = self._two_hot_ce(self._value_logits(phi_next), success)
            value_loss = value_loss + _masked_mean(aux_ce, next_is_terminal)

        value = self._value(logits)
        value_info = {
            "value_loss": value_loss,
            "value_mean": jnp.mean(value),
            # The separation between these two is the metric that says whether the head learned
            # anything beyond the marginal success rate.
            "value_mean_success": _masked_mean(value, success),
            "value_mean_failure": _masked_mean(value, 1.0 - success),
            "value_td_resid": jnp.mean(jnp.abs(value - target)),
            # How often the labeled terminal state is even present in a batch (~1/episode_length).
            "value_terminal_frac": jnp.mean(done),
            "value_at_terminal": _masked_mean(value, done),
        }
        total_loss = total_loss + self.value_loss_coeff * value_loss
        return total_loss, {**info, **value_info, "rep_loss": info["rep_loss"] + value_loss}
