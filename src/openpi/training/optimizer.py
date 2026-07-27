import dataclasses
from typing import Protocol, runtime_checkable

import flax.nnx as nnx
import jax.numpy as jnp
import optax

from openpi.shared import nnx_utils
import openpi.shared.array_typing as at

# Substrings marking params that decoupled weight decay should skip. Everything here is
# either 1-D or a single learned token: biases, LayerNorm/RMSNorm scales, positional and
# input embeddings, the rep heads' softmax layer-mix logits (`phi_mix`/`psi_mix`), the
# attention-pooling `query` token, and CRL's scalar `logit_scale` temperature. Shrinking
# these toward zero buys no regularization and actively fights what they are for — decaying
# a norm scale undoes the normalization, decaying the mix logits flattens the layer mix
# toward uniform. Decay weight matrices only.
NO_DECAY_PATH_REGEX = r".*(bias|scale|pos_embedding|input_embedding|query|_mix).*"


def kernel_decay_mask(no_decay_regex: str | None = NO_DECAY_PATH_REGEX):
    """Build an `optax.adamw` `mask` that decays weight matrices only.

    Returns a callable rather than a pytree because the optimizer is constructed before the
    params exist (see `scripts/train.py:init_train_state`). optax accepts either.

    The predicate mirrors the "kernel params" filter the train loop already uses for
    `param_norm`: ndim > 1 AND the path does not match `no_decay_regex`.
    """

    def mask(params: nnx.State) -> nnx.State:
        if no_decay_regex is None:
            decay_filter = lambda _, x: x.value.ndim > 1  # noqa: E731
        else:
            decay_filter = nnx.All(
                nnx.Not(nnx_utils.PathRegex(no_decay_regex)),
                lambda _, x: x.value.ndim > 1,
            )
        decay_keys = set(params.filter(decay_filter).flat_state())
        # `replace` keeps the VariableState wrapper, so the mask has exactly the same tree
        # structure as `params` with boolean leaves.
        return params.map(lambda k, v: v.replace(k in decay_keys))

    return mask


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Decoupled weight decay. The default is a negligible placeholder, NOT a regularizer —
    # changing it to exactly 0 can cause out-of-memory errors for some reason. To actually
    # regularize, set this to something real (1e-2 is a normal starting point) and make sure
    # a `weight_decay_mask` is supplied so it hits weight matrices only; see
    # `kernel_decay_mask` and `TrainConfig.weight_decay_exclude_regex`.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        # Accepted and ignored: plain SGD has no weight-decay term for a mask to apply to.
        # (The train loop builds one unconditionally, so rejecting it would break SGD runs.)
        del weight_decay_mask
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
