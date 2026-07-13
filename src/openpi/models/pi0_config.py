import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0
    from openpi.models.pi05rep import Pi0 as Pi0Rep


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    paligemma_lora_rank: int | None = None
    paligemma_lora_alpha: float | None = None
    action_expert_lora_rank: int | None = None
    action_expert_lora_alpha: float | None = None

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class Pi0RepConfig(Pi0Config):
    """Config for the CRL representation-learning variant of Pi0.

    Identical architecture to Pi0Config (pi05=True by default) but instantiates
    pi05rep.Pi0, which adds dedicated phi/psi self-attention heads that pool a
    learned mix over all backbone layers for contrastive RL training.
    """

    pi05: bool = True
    rep_dim: int = 512
    crl_loss_coeff: float = 0.01
    # Weight on the flow-matching action loss. 1.0 = normal joint training; 0.0
    # switches the action loss off entirely for pure representation learning. Note:
    # with action_loss_coeff=0.0 AND rep_backbone_grad_scale=0.0 the backbone gets
    # no gradient, so `get_freeze_filter` freezes it outright (see `backbone_frozen`):
    # only the phi/psi heads are trainable and the backbone's backward pass and
    # optimizer state are skipped. Raise rep_backbone_grad_scale to instead train
    # the backbone from the rep loss alone.
    action_loss_coeff: float = 1.0
    # Number of gemma blocks in each (phi/psi) representation head. Per-config
    # hyperparameter: raise for more head capacity, lower (e.g. 1) for leaner heads.
    rep_head_depth: int = 2
    # How much of the CRL representation loss gradient flows into the shared backbone.
    # 0.0 => full stop_gradient (backbone shaped only by the action loss; the rep heads
    # are read-only probes over a learned layer-mix). Values in (0, 1] scale the leak.
    rep_backbone_grad_scale: float = 0.0
    # Input the phi (current-time CRL anchor) representation is trained on:
    # "state_action" — phi token lives on the action suffix and sees the noisy
    # action tokens (Q-value style, the default); "state" — phi token joins psi
    # on the prefix (the state-only portion of the VLA backbone), making both
    # representations action-independent (state-value style). psi (the future-
    # time CRL target) is always state-input by design. The two rep tokens
    # never attend to each other.
    phi_input: str = "state_action"
    # Dropout applied inside the phi/psi rep heads (on the pooled query vector) during
    # training only. 0.0 = off (previous behavior). A regularizer for the small trainable
    # head set over a frozen backbone; helps close the train<<val rep-loss gap.
    rep_head_dropout: float = 0.0
    # CRL reps are L2-normalized before the InfoNCE dot products, so logits are cosine
    # similarities in [-1, 1]. A learnable temperature (initialized to this value, CLIP-style)
    # restores separability; without it normalized logits can't sharpen. Stored as a learnable
    # logit_scale = log(1/temperature); clamped at exp <= 100 to avoid runaway.
    crl_temperature_init: float = 0.07
    # Coefficient on the logsumexp penalty (guards against logit blow-up / collapse). With
    # L2-norm + temperature the logit scale is already bounded, so this defaults lower than the
    # old hardcoded 0.1. Set to 0.0 to disable.
    logsumexp_penalty_coeff: float = 0.01

    def __post_init__(self):
        super().__post_init__()
        if self.phi_input not in ("state_action", "state"):
            raise ValueError(
                f"phi_input must be 'state_action' or 'state', got {self.phi_input!r}"
            )

    @property
    def backbone_frozen(self) -> bool:
        """True when the shared backbone receives no gradient and can be frozen.

        This holds when the action loss is off (``action_loss_coeff == 0.0``) AND no
        rep-loss gradient leaks into the backbone (``rep_backbone_grad_scale == 0.0``).
        In that regime only the phi/psi rep heads are trainable, so the backbone's
        backward pass and optimizer state can be skipped entirely.
        """
        return self.action_loss_coeff == 0.0 and self.rep_backbone_grad_scale == 0.0

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        # When the backbone gets no gradient, freeze everything except the rep heads
        # (phi/psi head blocks, layer-mix logits, and output projections). This drops
        # the backbone from the trainable set, so its backward pass and optimizer
        # state are never materialized — a large speed and memory win.
        if self.backbone_frozen:
            # Rep-head params: phi/psi head blocks, layer-mix logits, output projections,
            # plus the learnable CRL temperature (logit_scale).
            rep_head_filter = nnx_utils.PathRegex(r".*((phi|psi)_(head|mix|proj)|logit_scale).*")
            return nnx.Not(rep_head_filter)
        return super().get_freeze_filter()

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Rep":
        from openpi.models.pi05rep import Pi0 as Pi0Rep

        return Pi0Rep(self, rngs=nnx.Rngs(rng))
