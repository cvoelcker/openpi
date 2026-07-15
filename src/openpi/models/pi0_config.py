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
    from openpi.models.pi05crl import Pi0CRL
    from openpi.models.pi05self_pred import Pi0SP
    from openpi.models.pi05sf import Pi0SF


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


_REP_INPUTS = ("state", "state_action")


@dataclasses.dataclass(frozen=True)
class Pi0RepBaseConfig(Pi0Config):
    """Shared config for the phi/psi representation-head variants (Pi0CRL, Pi0SF).

    Identical backbone to Pi0Config (pi05=True by default) plus dedicated phi/psi
    attention-pooling heads that pool a learned mix over all backbone layers
    (see rep_base.Pi0RepBase). Subclasses add their auxiliary loss.
    """

    pi05: bool = True
    # phi/psi representation dimension. Both reps project to rep_dim.
    rep_dim: int = 512
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
    # How much of the auxiliary (rep) loss gradient flows into the shared backbone.
    # 0.0 => full stop_gradient (backbone shaped only by the action loss; the rep heads
    # are read-only probes over a learned layer-mix). Values in (0, 1] scale the leak.
    rep_backbone_grad_scale: float = 0.0
    # Input each representation is trained on:
    # "state_action" — the rep head reads the action suffix and sees the (noisy)
    # action tokens (Q-value style); "state" — the rep head reads the prefix (the
    # state-only portion of the VLA backbone), making it action-independent
    # (state-value style). Defaults preserve the original behavior: phi (the
    # current-time anchor) is state_action, psi (the future-time target) is state.
    # The rep heads never attend to each other.
    phi_input: str = "state_action"
    psi_input: str = "state"
    # Dropout applied inside the phi/psi rep heads (on the pooled query vector) during
    # training only. 0.0 = off. A regularizer for the small trainable head set over a
    # frozen backbone; helps close the train<<val rep-loss gap.
    rep_head_dropout: float = 0.0
    # Whether to build the psi trio (psi_head/psi_mix/psi_proj) at all. Subclasses whose
    # loss never reads psi disable it to drop ~1e8 dead params and, in the
    # frozen-backbone regime, their optimizer state.
    enable_psi_head: bool = True
    # If set, psi is not trained by gradient descent; instead it is a lagging EMA copy of
    # phi (a BYOL/JEPA-style target network): psi <- ema * psi + (1 - ema) * phi after every
    # optimizer step. psi is initialized to phi and requires the same architecture
    # (psi_input == phi_input). TrainConfig.trainable_filter excludes the psi trio when this
    # is set, so it gets no gradients, no optimizer state, and stays float32 (never routed
    # through the bfloat16 freeze cast — EMA accumulation needs the precision).
    psi_lagging_ema: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.phi_input not in _REP_INPUTS:
            raise ValueError(f"phi_input must be one of {_REP_INPUTS}, got {self.phi_input!r}")
        if self.psi_input not in _REP_INPUTS:
            raise ValueError(f"psi_input must be one of {_REP_INPUTS}, got {self.psi_input!r}")
        if self.psi_lagging_ema is not None:
            if not self.enable_psi_head:
                raise ValueError("psi_lagging_ema requires enable_psi_head=True (psi is the lagging copy of phi)")
            if not 0.0 <= self.psi_lagging_ema < 1.0:
                raise ValueError(f"psi_lagging_ema must be in [0, 1), got {self.psi_lagging_ema}")
            if self.psi_input != self.phi_input:
                raise ValueError(
                    "psi_lagging_ema requires psi_input == phi_input (the lagging psi mirrors phi's "
                    f"architecture), got psi_input={self.psi_input!r}, phi_input={self.phi_input!r}"
                )

    @property
    @override
    def requires_goal_data(self) -> bool:
        return True

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
        # (phi/psi head blocks, layer-mix logits, and output projections, plus the
        # learnable CRL temperature logit_scale where present). This drops the backbone
        # from the trainable set, so its backward pass and optimizer state are never
        # materialized — a large speed and memory win. (Pi0SF's z_proj is excluded on
        # purpose: its gradient path runs through the stop-gradiented backbone, so it
        # cannot train in this regime anyway.)
        if self.backbone_frozen:
            rep_head_filter = nnx_utils.PathRegex(r".*((phi|psi)_(head|mix|proj)|logit_scale).*")
            return nnx.Not(rep_head_filter)
        return super().get_freeze_filter()


@dataclasses.dataclass(frozen=True)
class Pi0CRLConfig(Pi0RepBaseConfig):
    """Config for the contrastive-RL (CRL) representation-learning variant of Pi0.

    Instantiates pi05crl.Pi0CRL, which trains the phi/psi heads with a symmetric
    InfoNCE loss alongside the flow-matching action loss.
    """

    crl_loss_coeff: float = 0.01
    # CRL reps are L2-normalized before the InfoNCE dot products, so logits are cosine
    # similarities in [-1, 1]. A learnable temperature (initialized to this value, CLIP-style)
    # restores separability; without it normalized logits can't sharpen. Stored as a learnable
    # logit_scale = log(1/temperature); clamped at exp <= 100 to avoid runaway.
    crl_temperature_init: float = 0.07
    # Coefficient on the logsumexp penalty (guards against logit blow-up / collapse). With
    # L2-norm + temperature the logit scale is already bounded, so this defaults lower than the
    # old hardcoded 0.1. Set to 0.0 to disable.
    logsumexp_penalty_coeff: float = 0.01

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0CRL":
        from openpi.models.pi05crl import Pi0CRL

        return Pi0CRL(self, rngs=nnx.Rngs(rng))


@dataclasses.dataclass(frozen=True)
class Pi0SFConfig(Pi0RepBaseConfig):
    """Config for the TD Successor-Features variant of Pi0.

    Instantiates pi05sf.Pi0SF, which trains the phi/psi heads with a semi-gradient
    SARSA TD loss alongside the flow-matching action loss.
    """

    sf_gamma: float = 0.98  # TD discount
    fb_train_goal_ratio: float = 0.5  # fraction of the batch whose z = B(future); rest random
    # Weight on the TD successor-feature loss (Pi0CRLConfig's crl_loss_coeff analogue).
    sf_loss_coeff: float = 1.0
    # Weight on the psi orthonormality loss E[psi psi^T] ~ I (FB's ortho_coef analogue). This is
    # the anti-collapse term: with psi stop-gradded in the TD target, it is psi's ONLY gradient
    # source, so it must be > 0 to keep psi from collapsing to a constant. FB default 1.0.
    sf_ortho_coeff: float = 1.0
    # EMA decay for the phi(s',a') target network used in the TD bootstrap. None disables it
    # (bootstrap uses the online phi head, as before). A float in [0,1) EMAs a float32 copy of
    # the phi-head pathway (phi_mix/phi_head/phi_proj) toward the online weights each step; the
    # bootstrap then reads that slower-moving target. Standard TD-stability hygiene. Typical 0.99.
    sf_target_ema: float | None = None

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0SF":
        from openpi.models.pi05sf import Pi0SF

        return Pi0SF(self, rngs=nnx.Rngs(rng))


@dataclasses.dataclass(frozen=True)
class Pi0SPConfig(Pi0RepBaseConfig):
    """Config for the Self-Prediction variant of Pi0.

    Instantiates pi05self_pred.Pi0SP, which trains the phi head with a latent self-prediction
    loss (MSE + SigREP) alongside the flow-matching action loss. With the default
    psi_lagging_ema, psi is a frozen lagging (EMA) copy of phi that provides the
    self-prediction target — a BYOL/JEPA-style target network that stabilizes the bootstrap.
    With psi_lagging_ema=None the psi trio is not built and the target is the stop-gradient
    online phi (the original behavior); psi accessors then serve phi instead.
    """

    # EMA decay of the lagging psi target (see Pi0RepBaseConfig.psi_lagging_ema). None
    # disables the target network entirely: no psi trio, stop-gradient online targets.
    # enable_psi_head is derived from this in __post_init__ — do not set it directly.
    psi_lagging_ema: float | None = 0.995
    # phi must be action-independent ("state"): the actions enter the forward model
    # explicitly, concatenated to phi(s) in ForwardProjHead. A suffix ("state_action") phi
    # would leak the action into both the input rep and the prediction target.
    # Enforced in __post_init__.
    phi_input: str = "state"
    psi_input: str = "state"
    sp_loss_coeff: float = 1.0  # Weight on the self-prediction (MSE) loss
    forward_proj_blocks: int = 2  # Number of BRO blocks in the forward projection head
    # Weight on the SigREP loss — LeJEPA's SIGReg (sketched isotropic-Gaussian
    # regularization of the phi distribution), the anti-collapse complement to the MSE.
    # 0.0 disables it. Note the Epps-Pulley statistic is small (bounded, CF-scale), so this
    # coefficient typically wants to be >> the MSE's.
    sigrep_loss_coeff: float = 1.0
    # SIGReg sketch/quadrature: number of random 1D projections (resampled every step), and
    # the Epps-Pulley integration grid over t in [-sigrep_t_max, sigrep_t_max]. The N(0,1)
    # weight makes tails beyond |t|~4 negligible.
    sigrep_num_slices: int = 512
    sigrep_t_max: float = 4.0
    sigrep_num_t: int = 17

    def __post_init__(self):
        # psi exists exactly when it serves as the lagging target (SP has no other psi use).
        object.__setattr__(self, "enable_psi_head", self.psi_lagging_ema is not None)
        super().__post_init__()
        if self.phi_input != "state":
            raise ValueError(
                "Pi0SP requires phi_input='state': actions are concatenated to phi explicitly "
                f"in ForwardProjHead, got phi_input={self.phi_input!r}"
            )

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        # Same regime as Pi0RepBaseConfig, but forward_proj must stay trainable too: it takes
        # the (trainable) phi head's output, so it trains even with a frozen backbone.
        # SigREP itself has no learnable parameters.
        if self.backbone_frozen:
            trainable = nnx_utils.PathRegex(r".*((phi|psi)_(head|mix|proj)|forward_proj).*")
            return nnx.Not(trainable)
        return super().get_freeze_filter()

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0SP":
        from openpi.models.pi05self_pred import Pi0SP

        return Pi0SP(self, rngs=nnx.Rngs(rng))
