"""RLT-style query-token encoders trained with the Forward-Backward objective.

Self-contained on purpose: nothing outside this file is edited except a two-line splice in
`config.py` to register the arms, which is unavoidable because `_CONFIGS` is the registry
`get_config()` reads and `TrainConfig.freeze_filter` is `tyro.conf.Suppress`.

The representation architecture of RL Token (Xu, Springenberg, Equi, Amin, Esmail, Levine, Ke;
Physical Intelligence -- https://www.pi.website/download/rlt.pdf), with the objective swapped:
RLT trains z_rl = g_phi([z_{1:M}, e_rl])_{M+1} so a decoder can autoregressively reconstruct the
frozen VLA's token embeddings (paper Eq. 2); here the SAME encoder is trained by the FB bilinear
Bellman residual M(s,a,s') ~= F(s,a)^T B(s'), i.e. `pi05fb.fb_loss_terms`, unmodified.

Deliberate departures from the paper:

  * ENCODER WIDTH is a config field, not inherited from the VLA embedding width, so capacity can
    be traded against parameters. See the cost table below.
  * LANGUAGE TOKENS ARE KEPT. RLT drops them (fixed instruction per task); LIBERO-90 has 90
    instructions and B(s) must separate them.
  * F EXISTS AT ALL. RLT's single RL token is action-independent -- that is B. FB needs a
    state-action map, so a second encoder runs over the action-expert suffix.
  * NO PROPRIO IN THE ENCODER, matching RLT, which feeds it to the downstream actor/critic.
  * FINAL-LAYER EMBEDDINGS ONLY, no learned layer mix -- which also deletes a confound, since
    the mix measurably never moved in prior runs (psi_mix_entropy 2.72 -> 2.68 over 11,250 steps).

WHY d_fb IS CAPPED BY enc_width. For B = W_o z_B + b with z_B in R^W,

    E[B B^T] = W_o E[z_B z_B^T] W_o^T + W_o E[z_B] b^T + b E[z_B]^T W_o^T + b b^T
    ==> rank( E[B B^T] ) <= W + 1        at every batch size, for every input

so E[B B^T] = I_d is unreachable when d > W. Not a failure -- the ortho loss settles B onto an
orthonormal frame in a (W+1)-dim subspace -- but d_fb > enc_width buys nothing except wasted
`*_out` parameters and a `B_ortho_err` that never converges. The same holds for F for a different
reason: Q_r(s,a) = F(s,a)^T z_r depends only on z_r's projection onto span(F), so a rank-r F can
express only an r-dimensional reward space. Both encoders must be at least d_fb wide.

COST TABLE (both encoders, trainable only). Per block: 4*W^2 + 2*W*M, quadratic in width.

    enc_width  depth  mlp_dim   d_fb   trainable    note
      768        3     3072      768     46.8 M     fits a 30-50M budget
     1024        2     4096     1024     55.6 M
     2048        2     4096     2048    151.0 M     measured
     2048        3     4096     2048    218.2 M     <-- DEFAULT for pi05_fb_rlt_onpolicy
     2048        3     8192     2048    302.0 M     do not

The model logs its exact parameter count at construction (`rlt/params_total`), so the table is a
guide, not a contract.
"""

import dataclasses
import logging
from typing import Any, Literal

import flax.linen as nn
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.models import rep_grad_guidance as _rgg
from openpi.models.pi05fb import _FBTarget
from openpi.models.pi05fb import fb_loss_terms
from openpi.models.rep_grad_guidance import ValueGuidedSamplingMixin
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

logger = logging.getLogger("openpi")


# =====================================================================================
# Diagnostics. Everything here is stop-gradded and cheap; see the module docstring's
# logging contract. Prefixes: `rep/` per-representation, `enc/` encoder internals,
# `fb/` objective, `opt/` optimization health.
# =====================================================================================


def _effective_rank(x: at.Array, eps: float = 1e-12) -> at.Array:
    """Participation ratio of the centered covariance's eigenvalues: (sum L)^2 / sum L^2.

    The metric to read when rep_dim > batch_size, where `B_ortho_err` is floored at (d-b)/d and
    says nothing. Computed on the b x b Gram (same nonzero spectrum, far cheaper at d=2048) with
    no eigendecomposition, since sum L = tr(G) and sum L^2 = ||G||_F^2 for symmetric G.

    Bounded above by min(b - 1, d): pinned at b-1 means batch-limited, not representation-limited.
    """
    x = x.astype(jnp.float32)
    xc = x - jnp.mean(x, axis=0, keepdims=True)
    g = xc @ xc.T
    return jnp.square(jnp.trace(g)) / (jnp.sum(jnp.square(g)) + eps)


def _rep_health(x: at.Array, name: str) -> dict[str, at.Array]:
    """Full health panel for one [b, d] representation.

    Read together, not individually:
      norm_mean + feat_std near 0        -> collapse to a point
      feat_std low but mean_norm high    -> collapse to a CONSTANT (the failure that uncentered
                                            metrics reward; see pi05sf.py's centered companions)
      pairwise_cos_abs -> 1              -> directional collapse
      eff_rank_frac -> 0                 -> using a small subspace of the d available
      dead_frac -> 1                     -> individual channels have died
    """
    x = jax.lax.stop_gradient(x.astype(jnp.float32))
    b, d = x.shape
    norms = jnp.linalg.norm(x, axis=-1)
    xn = x / (norms[:, None] + 1e-8)
    cos = xn @ xn.T
    off = 1.0 - jnp.eye(b, dtype=cos.dtype)
    n_off = float(max(b * (b - 1), 1))
    feat_std = jnp.std(x, axis=0)
    eff = _effective_rank(x)
    return {
        f"rep/{name}/norm_mean": jnp.mean(norms),
        f"rep/{name}/norm_std": jnp.std(norms),
        # Across-batch per-feature std, meaned over features. -> 0 as the rep becomes constant.
        f"rep/{name}/feat_std": jnp.mean(feat_std),
        f"rep/{name}/feat_std_min": jnp.min(feat_std),
        # Norm of the batch mean. A large constant component hides under a healthy feat_std.
        f"rep/{name}/mean_norm": jnp.linalg.norm(jnp.mean(x, axis=0)),
        f"rep/{name}/eff_rank": eff,
        # Normalized by the CEILING the batch can express, so 1.0 means "as spread as b allows".
        f"rep/{name}/eff_rank_frac": eff / float(min(b - 1, d)),
        f"rep/{name}/pairwise_cos_abs": jnp.sum(jnp.abs(cos) * off) / n_off,
        f"rep/{name}/dead_frac": jnp.mean((feat_std < 1e-4).astype(jnp.float32)),
        f"rep/{name}/max_abs": jnp.max(jnp.abs(x)),
        # Non-finite counts. Cheap, and the difference between a silent NaN and a lost run.
        f"rep/{name}/nonfinite_frac": jnp.mean((~jnp.isfinite(x)).astype(jnp.float32)),
    }


def _entropy_from_probs(p: at.Array, axis: int = -1) -> at.Array:
    return -jnp.sum(p * jnp.log(p + 1e-9), axis=axis)


# =====================================================================================
# The encoder
# =====================================================================================


class _RLTBlock(nn.Module):
    """Pre-LN transformer encoder block with an explicit (observable) attention computation.

    Written out rather than delegated to `nn.MultiHeadDotProductAttention`, which does not return
    the attention weights -- and the RL-token query row is the most informative signal about
    whether the readout is doing anything (`enc/*/rl_attn_*`). Parameter count is identical.
    `siglip.Encoder1DBlock` takes no mask at all, and our prefix has masked tokens.
    """

    width: int
    mlp_dim: int
    num_heads: int
    dtype_mm: str

    @nn.compact
    def __call__(self, x, attn_mask, deterministic: bool = True):  # noqa: FBT001, FBT002
        head_dim = self.width // self.num_heads
        y = nn.LayerNorm(dtype=self.dtype_mm, name="ln_attn")(x)

        proj = lambda name: nn.DenseGeneral(  # noqa: E731
            features=(self.num_heads, head_dim),
            axis=-1,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
            name=name,
        )
        q, k, v = proj("q")(y), proj("k")(y), proj("v")(y)

        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32)
        logits = logits / jnp.sqrt(jnp.asarray(head_dim, jnp.float32))
        # Finite, not -inf: an all-masked row under -inf gives 0/0 = NaN from softmax, whereas
        # equal finite logits give a harmless uniform row. Same reasoning as gemma.Attention.
        logits = jnp.where(attn_mask, logits, -2.3819763e38)
        attn = jax.nn.softmax(logits, axis=-1)  # [b, h, q, k], float32

        out = jnp.einsum("bhqk,bkhd->bqhd", attn.astype(v.dtype), v)
        out = nn.DenseGeneral(
            features=self.width,
            axis=(-2, -1),
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
            name="out",
        )(out)
        x = x + out

        y = nn.LayerNorm(dtype=self.dtype_mm, name="ln_mlp")(x)
        y = nn.Dense(self.mlp_dim, dtype=self.dtype_mm, kernel_init=nn.initializers.xavier_uniform(), name="mlp_in")(y)
        y = nn.gelu(y)
        y = nn.Dense(self.width, dtype=self.dtype_mm, kernel_init=nn.initializers.xavier_uniform(), name="mlp_out")(y)
        x = x + y

        # Everything below is stop-gradded so the diagnostics can never perturb the loss graph.
        sg = jax.lax.stop_gradient
        stats = {
            # The RL token is the LAST query row. Its attention distribution is the readout.
            #   entropy -> log(T)  : the query is averaging (a mask-mean in disguise)
            #   entropy -> 0       : it has collapsed onto one token
            # Somewhere in between is what selection looks like.
            "rl_attn_entropy": sg(jnp.mean(_entropy_from_probs(attn[:, :, -1, :]))),
            # Attention mass the RL token spends on ITSELF rather than on the memory. -> 1 means
            # the readout has stopped reading the sequence entirely.
            "rl_attn_self": sg(jnp.mean(attn[:, :, -1, -1])),
            "rl_attn_max": sg(jnp.mean(jnp.max(attn[:, :, -1, :], axis=-1))),
            # Averaged over ALL query rows, for comparison against the RL row specifically.
            "attn_entropy_all": sg(jnp.mean(_entropy_from_probs(attn))),
            # Residual-stream magnitude leaving this block. Pre-LN streams grow monotonically;
            # a sudden jump or decay to zero is the first sign of an unstable depth.
            "resid_norm": sg(jnp.mean(jnp.linalg.norm(x.astype(jnp.float32), axis=-1))),
            "attn_out_norm": sg(jnp.mean(jnp.linalg.norm(out.astype(jnp.float32), axis=-1))),
            "mlp_out_norm": sg(jnp.mean(jnp.linalg.norm(y.astype(jnp.float32), axis=-1))),
        }
        return x, stats


class _RLTHead(nn.Module):
    """RLT's query-token readout: project -> +posemb -> append e_rl -> encoder -> read e_rl's row.

    Call contract matches the other heads in this repo -- (b, t, d_in) memory plus a (b, t) bool
    mask in, one (b, enc_width) vector out -- so it is a drop-in for `_MeanPoolRepHead`.
    """

    enc_width: int
    depth: int
    mlp_dim: int
    num_heads: int
    max_tokens: int
    embed_dtype: str
    dropout: float = 0.0
    remat: bool = False

    @nn.compact
    def __call__(self, memory, memory_mask, *, deterministic: bool = True, dropout_rng=None):
        b, t, _ = memory.shape
        x = nn.Dense(
            self.enc_width, dtype=self.embed_dtype, kernel_init=nn.initializers.xavier_uniform(), name="in_proj"
        )(memory.astype(self.embed_dtype))
        in_proj_norm = jnp.mean(jnp.linalg.norm(x.astype(jnp.float32), axis=-1))

        # A fresh encoder is permutation-equivariant over its inputs. The VLA's RoPE shaped the
        # embeddings' CONTENT but carries no usable position signal into a new attention stack.
        pos = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0 / np.sqrt(self.enc_width)),
            (1, self.max_tokens, self.enc_width),
        )
        x = x + pos[:, :t].astype(x.dtype)

        e_rl = self.param("rl_token", nn.initializers.normal(0.02), (1, 1, self.enc_width))
        x = jnp.concatenate([x, jnp.broadcast_to(e_rl, (b, 1, self.enc_width)).astype(x.dtype)], axis=1)
        seq_mask = jnp.concatenate([memory_mask, jnp.ones((b, 1), dtype=jnp.bool_)], axis=1)
        # Bidirectional among valid tokens -- this is an encoder, not a decoder. [b, 1, T, T].
        attn_mask = (seq_mask[:, None, :] & seq_mask[:, :, None])[:, None]

        # A Python loop rather than nn.scan: depth is small (1-3), and this gives per-block
        # parameter names and per-block diagnostics without fighting scan's carry/stacking.
        #
        # Remat is a speed/memory trade and memory is not binding at this batch size: retained
        # activations are ~2 GB per block per 32 examples at W=2048, against a 96 GB GH200. Turn
        # it back on if you raise per-device batch or depth and hit an OOM; correctness is
        # identical either way.
        block_cls = (
            nn.remat(
                _RLTBlock,
                prevent_cse=False,
                static_argnums=(3,),
                policy=jax.checkpoint_policies.nothing_saveable,
            )
            if self.remat
            else _RLTBlock
        )
        stats: dict[str, at.Array] = {"in_proj_norm": in_proj_norm}
        for i in range(self.depth):
            x, blk = block_cls(
                width=self.enc_width,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dtype_mm=self.embed_dtype,
                name=f"block_{i}",
            )(x, attn_mask, deterministic)
            stats |= {f"b{i}/{k}": v for k, v in blk.items()}

        x = nn.LayerNorm(dtype=self.embed_dtype, name="encoder_norm")(x)
        z = x[:, -1]  # the RL token

        # Scale of the learned positional table relative to the projected tokens. If posemb_norm
        # dwarfs in_proj_norm the encoder is attending to position rather than content.
        stats["posemb_norm"] = jax.lax.stop_gradient(jnp.mean(jnp.linalg.norm(pos[0, :t].astype(jnp.float32), axis=-1)))
        stats["rl_token_norm"] = jax.lax.stop_gradient(jnp.linalg.norm(e_rl.astype(jnp.float32)))
        stats["valid_token_frac"] = jax.lax.stop_gradient(jnp.mean(seq_mask.astype(jnp.float32)))

        if self.dropout > 0.0 and not deterministic and dropout_rng is not None:
            keep = 1.0 - self.dropout
            mask = jax.random.bernoulli(dropout_rng, keep, z.shape)
            z = jnp.where(mask, z / keep, 0.0)
        return z, stats


# =====================================================================================
# Config
# =====================================================================================


@dataclasses.dataclass(frozen=True)
class Pi0FBRLTConfig(pi0_config.Pi0Config):
    """FB objective on RLT-style query-token encoders. See the module docstring."""

    # --- inherited-but-load-bearing -------------------------------------------------
    pi05: bool = True
    # Never inherit this. Pi0Config defaults it to `pi05`, i.e. True, which tokenizes the state
    # into the PROMPT -- changing the frozen backbone's input distribution away from what
    # pi05_libero was fine-tuned under (config.py sets it False) and destroying the
    # `guidance scale 0.0 == base policy` invariant every paired comparison here rests on.
    discrete_state_input: bool = False

    # --- encoder geometry (see the cost table in the module docstring) ---------------
    rlt_enc_width: int = 2048
    rlt_depth: int = 2
    rlt_mlp_dim: int = 4096
    rlt_num_heads: int = 16
    rlt_dropout: float = 0.0  # RLT has NO encoder dropout; its "dropout" is reference-action dropout
    # Gradient checkpointing inside the encoder blocks. False is FASTER (no backward recompute)
    # and is affordable at small per-device batch; see _RLTHead for the memory arithmetic.
    rlt_remat: bool = False
    # Separate knobs for F, which reads 10 tokens against B's 968. None => mirror B.
    rlt_f_enc_width: int | None = None
    rlt_f_depth: int | None = None

    # --- FB objective ---------------------------------------------------------------
    rep_dim: int = 2048  # d_fb. MUST be <= min(enc_width_B, enc_width_F); validated below.
    fb_gamma: float = 0.98
    # L2-normalize B onto the unit sphere (SF's `sf_normalize_psi`). See Pi0FBRLT._finalize for
    # the measurements that motivated it and for what it costs. F is never normalized.
    fb_normalize_B: bool = True  # noqa: N815 -- matches this file's F_/B_ head naming
    # None => 1.0 when fb_normalize_B, else 1 / rep_dim. A float pins it explicitly.
    #
    # Unnormalized, l_ortho's off-diagonal term is ~||B||^4 ~ d^2 (~1e6 at d=768), so it needs 1/d
    # to be commensurable with l_fb. Not the FB reference default of 1.0: a 20-step canary at
    # d=2048 gave fb_ortho_loss 2.49e6 against fb_loss 188 and B collapsed to eff_rank 3.76 of
    # 2048, global-norm clipping having scaled the whole update to nothing. This breaks numerical
    # comparability with published `orth_coef` values; `fb_loss_terms` itself is untouched.
    fb_ortho_coeff: float | None = None
    fb_target_ema: float | None = 0.99
    fb_quadratic_offdiag_only: bool = False
    rep_backbone_grad_scale: float = 0.0

    F_input: Literal["state", "state_action"] = "state_action"
    B_input: Literal["state", "state_action"] = "state"

    # --- deployment -----------------------------------------------------------------
    guidance_schedule: _rgg.GuidanceWeightSchedule = dataclasses.field(default_factory=_rgg.GuidanceWeightSchedule)
    guidance_grad_clip_norm: float | None = 10.0
    guidance_eval_at_xt: bool = False
    normalize_z_r: bool = True

    def __post_init__(self):
        # MUST come first: Pi0Config.__post_init__ fills max_token_len (200 for pi05). Shadowing
        # it without super() leaves the field at its None sentinel and the failure surfaces far
        # away as a shape error out of fake_obs().
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi0FBRLT requires pi05=True (suffix = action tokens only)")
        if self.B_input != "state":
            raise ValueError(f"B must be action-independent; got B_input={self.B_input!r}")
        f_w = self.rlt_f_enc_width or self.rlt_enc_width
        if self.rep_dim > min(self.rlt_enc_width, f_w):
            raise ValueError(
                f"rep_dim={self.rep_dim} exceeds an encoder width "
                f"(B={self.rlt_enc_width}, F={f_w}). rank(E[BB^T]) <= enc_width + 1, so the "
                "orthonormality target is structurally unreachable, and Q_r = F^T z_r can only "
                "see the span of F. See the module docstring."
            )
        for w, h, tag in ((self.rlt_enc_width, self.rlt_num_heads, "B"), (f_w, self.rlt_num_heads, "F")):
            if w % h:
                raise ValueError(f"{tag} enc_width {w} not divisible by rlt_num_heads {h}")

    @property
    @override
    def requires_goal_data(self) -> bool:
        """Routes training through rl_data_loader, which supplies next_observation /
        next_actions / next_is_pad."""
        return True

    @property
    def backbone_frozen(self) -> bool:
        return self.rep_backbone_grad_scale == 0.0

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Train only the F/B encoder + projection trios. No `mix` -- this model reads the final
        layer only, so there are no layer-mix logits to train."""
        if self.backbone_frozen:
            return nnx.Not(nnx_utils.PathRegex(r".*(F|B)_(head|out).*"))
        return super().get_freeze_filter()

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0FBRLT":
        return Pi0FBRLT(self, rngs=nnx.Rngs(rng))


# =====================================================================================
# Model
# =====================================================================================


class Pi0FBRLT(ValueGuidedSamplingMixin, _pi0.Pi0):
    """RLT encoders on a frozen pi0.5, trained by the FB bilinear objective."""

    def __init__(self, config: Pi0FBRLTConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.F_input = config.F_input
        self.B_input = config.B_input
        # Plain attributes, NOT properties. Pi0.__init__ assigns phi_input/psi_input for its own
        # mean-pooled steering interface, and nnx's __setattr__ raises if a subclass shadows them
        # with a read-only property. Overwriting them is also what makes
        # ValueGuidedSamplingMixin.test_time_adapt work unmodified -- it gates on
        # psi_input == "state" before building z_r, and B is this model's psi.
        self.psi_input = config.B_input
        self.phi_input = config.F_input

        self.rep_dim = config.rep_dim
        self.rep_backbone_grad_scale = config.rep_backbone_grad_scale
        self.fb_gamma = config.fb_gamma
        self.normalize_B = config.fb_normalize_B
        # Resolved once here so every consumer (the loss AND the gradient-balance diagnostic)
        # sees the same number; see Pi0FBRLTConfig.fb_ortho_coeff for the derivation.
        if config.fb_ortho_coeff is not None:
            self.fb_ortho_coeff = config.fb_ortho_coeff
        else:
            self.fb_ortho_coeff = 1.0 if config.fb_normalize_B else 1.0 / config.rep_dim
        self.fb_target_ema = config.fb_target_ema
        self.fb_quadratic_offdiag_only = config.fb_quadratic_offdiag_only
        self.guidance_schedule = config.guidance_schedule
        self.guidance_grad_clip_norm = config.guidance_grad_clip_norm
        self.guidance_eval_at_xt = config.guidance_eval_at_xt
        self.normalize_z_r_default = config.normalize_z_r

        widths = {"state": paligemma_config.width, "state_action": action_expert_config.width}
        b_in, f_in = widths[config.B_input], widths[config.F_input]
        b_w = config.rlt_enc_width
        f_w = config.rlt_f_enc_width or config.rlt_enc_width
        f_depth = config.rlt_f_depth if config.rlt_f_depth is not None else config.rlt_depth

        # Prefix length is images + language; suffix is the action chunk. `max_tokens` sizes the
        # positional table, so it must be an upper bound, not the exact length.
        # Spec-only (ShapeDtypeStructs), so this allocates nothing.
        n_img = len(config.inputs_spec()[0].images)
        b_max_tokens = n_img * 256 + config.max_token_len
        f_max_tokens = config.action_horizon

        def _head(enc_width, depth, max_tokens, mem_width):
            head = nnx_bridge.ToNNX(
                _RLTHead(
                    enc_width=enc_width,
                    depth=depth,
                    mlp_dim=config.rlt_mlp_dim,
                    num_heads=config.rlt_num_heads,
                    max_tokens=max_tokens,
                    embed_dtype=config.dtype,
                    dropout=config.rlt_dropout,
                    remat=config.rlt_remat,
                )
            )
            head.lazy_init(jnp.zeros((1, 2, mem_width)), jnp.ones((1, 2), dtype=bool), rngs=rngs)
            return head

        # B first, then F, so the rng draw order is stable if F's knobs later change.
        self.B_head = _head(b_w, config.rlt_depth, b_max_tokens, b_in)
        self.F_head = _head(f_w, f_depth, f_max_tokens, f_in)
        self.B_out = nnx.Linear(b_w, config.rep_dim, rngs=rngs)
        self.F_out = nnx.Linear(f_w, config.rep_dim, rngs=rngs)

        if self.fb_target_ema is not None:
            self._build_targets()

        self._log_param_summary(config)

    # ---------------------------------------------------------------------------------
    # Startup logging -- the parameter count is the first thing to check against the
    # cost table, and it is cheap to get wrong by a factor of 3.
    # ---------------------------------------------------------------------------------

    def _log_param_summary(self, config: Pi0FBRLTConfig) -> None:
        def count(mod) -> int:
            return int(sum(np.prod(v.value.shape) for v in nnx.state(mod, nnx.Param).flat_state().values()))

        parts = {
            "B_head": count(self.B_head),
            "B_out": count(self.B_out),
            "F_head": count(self.F_head),
            "F_out": count(self.F_out),
        }
        total = sum(parts.values())
        logger.info(
            "Pi0FBRLT trainable parameters: %s  TOTAL=%.2fM  normalize_B=%s ortho_coeff=%.4g  "
            "(enc_width B=%d F=%d, depth B=%d F=%d, mlp_dim=%d, heads=%d, d_fb=%d)",
            {k: f"{v / 1e6:.2f}M" for k, v in parts.items()},
            total / 1e6,
            self.normalize_B,
            self.fb_ortho_coeff,
            config.rlt_enc_width,
            config.rlt_f_enc_width or config.rlt_enc_width,
            config.rlt_depth,
            config.rlt_f_depth if config.rlt_f_depth is not None else config.rlt_depth,
            config.rlt_mlp_dim,
            config.rlt_num_heads,
            config.rep_dim,
        )
        self._param_counts = parts

    # ---------------------------------------------------------------------------------
    # Representations
    # ---------------------------------------------------------------------------------

    def _scale_grad(self, x, scale: float | None = None):
        """0 => full stop_gradient (cleanest graph); 1 => identity; else a partial pass whose
        forward value is unchanged and whose backward is scaled by `scale`."""
        scale = self.rep_backbone_grad_scale if scale is None else scale
        if scale == 0.0:
            return jax.lax.stop_gradient(x)
        if scale == 1.0:
            return x
        return scale * x + (1.0 - scale) * jax.lax.stop_gradient(x)

    def _rep(
        self,
        which: Literal["F", "B"],
        prefix_out,
        suffix_out,
        prefix_mask,
        suffix_mask,
        *,
        deterministic: bool = True,
        dropout_rng=None,
        grad_scale_override: float | None = None,
    ):
        """Final-layer embeddings -> RLT encoder -> linear projection to R^rep_dim.

        Returns (rep, z, stats): `rep` is the FB-space vector the loss consumes, `z` is the raw
        RL token (the thing downstream online RL would read, per RLT), `stats` is the encoder
        diagnostics panel.

        `grad_scale_override=1.0` is LOAD-BEARING at inference: `rep_backbone_grad_scale=0.0`
        otherwise puts a stop_gradient on the only path from `a` to F, so `grad_a_q` returns
        exactly zeros and guided sampling silently degenerates to unguided at every scale.
        """
        source = self.F_input if which == "F" else self.B_input
        mem, mask = (prefix_out, prefix_mask) if source == "state" else (suffix_out, suffix_mask)
        if mem is None:
            seq = "prefix" if source == "state" else "suffix"
            raise ValueError(f"{which}_input={source!r} requires {seq} output embeddings")
        # NOTE: no layer mix -- final-layer embeddings only, per RLT.
        mem = self._scale_grad(mem, grad_scale_override)
        head = self.F_head if which == "F" else self.B_head
        out = self.F_out if which == "F" else self.B_out
        z, stats = head(mem, mask, deterministic=deterministic, dropout_rng=dropout_rng)
        return self._finalize(which, out(z.astype(jnp.float32))), z.astype(jnp.float32), stats

    def _finalize(self, which: Literal["F", "B"], rep):
        """L2-normalize B onto the unit sphere; leave F alone. Only B's direction carries meaning,
        while F's magnitude is part of the quantity being predicted.

        Why: across two 2000-step runs whose ortho coefficients differ by 77x, B's per-dimension
        variance settled at 0.036 and 0.017 against a target of 1.0 either way. The off-diagonal
        penalty is ~||B||^4 while the diagonal reward is -2||B||^2, so while rows stay correlated
        the cheapest way to cut the penalty is to SHRINK B rather than decorrelate it -- and a
        larger coefficient makes shrinking more attractive. Normalizing removes that escape route.

        What it costs: `fb/B_sigma2` becomes the constant 1/d and carries no information, and the
        ortho diagonal term becomes a constant -2, leaving decorrelation as the whole objective.
        Acceptable, since decorrelation is the job the ortho term demonstrably can do (B_eff_rank
        3.9 -> 8.4 and rising) and scale is the one it cannot. Downstream the lost scale is
        harmless: `normalize_z_r=True` divides it out, and grad_a Q's direction is scale-invariant.
        """
        rep = rep.astype(jnp.float32)
        if which == "B" and self.normalize_B:
            rep = rep / (jnp.linalg.norm(rep, axis=-1, keepdims=True) + 1e-6)
        return rep

    # ---------------------------------------------------------------------------------
    # EMA targets (same pattern as pi05fb.py)
    # ---------------------------------------------------------------------------------

    @staticmethod
    def _relkey(relpath) -> str:
        """Target leaves live in STRING-keyed dicts: a list container makes nnx emit integer path
        components, which State.to_pure_dict() -> flatten_dict(sep='/') cannot join, crashing
        weight loading at init."""
        return "|".join(map(str, relpath))

    def _snapshot(self, module):
        flat = nnx.state(module).flat_state()
        relpaths = list(flat.keys())
        return {self._relkey(p): _FBTarget(jnp.asarray(flat[p].value, jnp.float32)) for p in relpaths}, relpaths

    def _build_targets(self):
        self._B_head_t, self._B_head_rp = self._snapshot(self.B_head)
        self._B_out_t, self._B_out_rp = self._snapshot(self.B_out)
        self._F_head_t, self._F_head_rp = self._snapshot(self.F_head)
        self._F_out_t, self._F_out_rp = self._snapshot(self.F_out)

    def _merge_target(self, online_module, targets, relpaths):
        """.replace preserves each leaf's exact VariableState type -- a fresh nnx.Param does not
        round-trip through the ToNNX bridge."""
        flat = nnx.state(online_module).flat_state()
        swapped = {p: flat[p].replace(targets[self._relkey(p)].value) for p in relpaths}
        return nnx.merge(nnx.graphdef(online_module), nnx.State.from_flat_path(swapped))

    def _target_rep(self, which, prefix_out, suffix_out, prefix_mask, suffix_mask):
        source = self.F_input if which == "F" else self.B_input
        mem, mask = (prefix_out, prefix_mask) if source == "state" else (suffix_out, suffix_mask)
        head = self._merge_target(
            getattr(self, f"{which}_head"), getattr(self, f"_{which}_head_t"), getattr(self, f"_{which}_head_rp")
        )
        out = self._merge_target(
            getattr(self, f"{which}_out"), getattr(self, f"_{which}_out_t"), getattr(self, f"_{which}_out_rp")
        )
        z, _ = head(self._scale_grad(mem), mask, deterministic=True, dropout_rng=None)
        # MUST apply the same normalization as _rep: the bootstrap B-bar is contracted against
        # F-bar in the very same bilinear form, so an unnormalized target would put the two
        # halves of the Bellman residual in different spaces.
        return self._finalize(which, out(z.astype(jnp.float32)))

    def _blend(self, decay: float):
        if self.fb_target_ema is None:
            return
        for which in ("F", "B"):
            for kind in ("head", "out"):
                online = nnx.state(getattr(self, f"{which}_{kind}")).flat_state()
                targets = getattr(self, f"_{which}_{kind}_t")
                for rp in getattr(self, f"_{which}_{kind}_rp"):
                    t = targets[self._relkey(rp)]
                    t.value = decay * t.value + (1.0 - decay) * online[rp].value.astype(jnp.float32)

    def sync_target_networks(self):
        """Called once by the train loop after the checkpoint merge."""
        self._blend(0.0)

    def update_target_networks(self):
        """Called every step after the optimizer update."""
        self._blend(self.fb_target_ema if self.fb_target_ema is not None else 0.0)

    # ---------------------------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------------------------

    def _forward_out(self, observation, actions, timestep):
        """Like Pi0._forward but keeps `prefix_out` (the post-final-norm prefix embeddings),
        which the base implementation discards. RLT reads final-layer embeddings, so we need it.
        Written here rather than editing pi0.py."""
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar, adarms = self.embed_suffix(observation, actions, timestep)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _, _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms],
        )
        return prefix_out, suffix_out, prefix_mask, suffix_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, batch: dict[str, Any], *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """FB loss on RLT encoders. No flow-matching action loss: this model never trains the
        policy, and a BC loss multiplied by zero costs a whole extra forward pass.

        Required batch keys: observation, actions, next_observation, next_actions.
        Optional: next_is_pad.
        """
        _model.require_batch_keys(
            batch, ("observation", "actions", "next_observation", "next_actions"), type(self).__name__
        )
        actions, next_actions = batch["actions"], batch["next_actions"]
        next_is_pad = batch.get("next_is_pad")

        pre_rng, next_pre_rng, drop_rng = jax.random.split(rng, 3)
        f_drop, b_drop = jax.random.split(drop_rng)
        deterministic = not train
        # Augment current/next independently so F(s,a) and the bootstrap cannot agree on shared
        # augmentation artifacts.
        observation = _model.preprocess_observation(pre_rng, batch["observation"], train=train)
        next_observation = _model.preprocess_observation(next_pre_rng, batch["next_observation"], train=train)

        b = actions.shape[0]
        obs_stacked = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), observation, next_observation
        )
        actions_stacked = jnp.concatenate([actions, next_actions], axis=0)
        time0 = jnp.zeros((actions_stacked.shape[0],), dtype=jnp.float32)  # clean actions: x_t == a
        prefix_out, suffix_out, prefix_mask, suffix_mask = self._forward_out(obs_stacked, actions_stacked, time0)

        # NOTE: these are 3-D (b, t, d), NOT the 4-D (L, b, t, d) hidden stacks. Slice axis 0.
        cur, nxt = slice(None, b), slice(b, None)

        def _slice(which, sl, *, det, rng_):
            return self._rep(
                which,
                prefix_out[sl],
                suffix_out[sl],
                prefix_mask[sl],
                suffix_mask[sl],
                deterministic=det,
                dropout_rng=rng_,
            )

        # F(s,a) from the current half's suffix; B(s') from the next half's prefix. The b next
        # states double as the x ~ rho sample the quadratic term integrates over, which is what
        # makes M a b x b matrix rather than requiring a separate draw.
        f_cur, z_f, f_stats = _slice("F", cur, det=deterministic, rng_=f_drop)
        b_next, z_b, b_stats = _slice("B", nxt, det=deterministic, rng_=b_drop)

        if self.fb_target_ema is not None:
            f_boot = self._target_rep("F", prefix_out[nxt], suffix_out[nxt], prefix_mask[nxt], suffix_mask[nxt])
            b_boot = self._target_rep("B", prefix_out[nxt], suffix_out[nxt], prefix_mask[nxt], suffix_mask[nxt])
        else:
            f_boot = _slice("F", nxt, det=True, rng_=None)[0]
            b_boot = _slice("B", nxt, det=True, rng_=None)[0]
        f_boot, b_boot = jax.lax.stop_gradient(f_boot), jax.lax.stop_gradient(b_boot)

        not_terminal = (
            jnp.ones((b,), dtype=f_cur.dtype) if next_is_pad is None else 1.0 - next_is_pad.astype(f_cur.dtype)
        )

        l_fb, l_ortho, metrics = fb_loss_terms(
            f_cur,
            b_next,
            f_boot,
            b_boot,
            gamma=self.fb_gamma,
            not_terminal=not_terminal,
            quadratic_offdiag_only=self.fb_quadratic_offdiag_only,
        )
        total = l_fb + self.fb_ortho_coeff * l_ortho

        info = {
            "fb_loss": l_fb,
            "fb_ortho_loss": l_ortho,
            "rep_loss": l_fb,  # alias so shared logging that expects the CRL/SF key set keeps working
            **{f"fb/{k}": v for k, v in metrics.items()},
            **_rep_health(f_cur, "F"),
            **_rep_health(b_next, "B"),
            # The RAW RL tokens, pre-projection. This is what downstream online RL would consume
            # (RLT's z_rl), so its health matters independently of the FB-space projection: B_out
            # can look fine while z_B has already collapsed, or vice versa.
            **_rep_health(z_f, "z_F"),
            **_rep_health(z_b, "z_B"),
            **{f"enc/F/{k}": v for k, v in f_stats.items()},
            **{f"enc/B/{k}": v for k, v in b_stats.items()},
            **self._cross_metrics(f_cur, b_next, f_boot, b_boot, z_f, z_b, not_terminal),
        }
        # train.py means this; broadcasting keeps the (*b, ah) shape the BaseModel contract
        # declares without inventing a per-timestep decomposition the FB loss does not have.
        return jnp.broadcast_to(total, (b, self.action_horizon)), info

    def _cross_metrics(self, f_cur, b_next, f_boot, b_boot, z_f, z_b, not_terminal) -> dict[str, at.Array]:
        """Relationships between the pieces, and the things the standard FB panel misses."""
        sg = jax.lax.stop_gradient
        f, bn, fb_, bb_, zf, zb = (sg(x.astype(jnp.float32)) for x in (f_cur, b_next, f_boot, b_boot, z_f, z_b))
        b = f.shape[0]

        def _cos_rows(p, q):
            return jnp.mean(
                jnp.sum(p * q, axis=-1) / (jnp.linalg.norm(p, axis=-1) * jnp.linalg.norm(q, axis=-1) + 1e-8)
            )

        m = f @ bn.T
        # Singular spectrum of the b x b bilinear form. Once d_fb >= b, F B^T can express ANY
        # b x b matrix, so a rank-1 M means the model found a degenerate way to shrink the
        # residual rather than a real factorization.
        sv = jnp.linalg.svd(m, compute_uv=False)
        return {
            # Matched-pair alignment: F(s,a) against B(s') for the SAME transition, versus
            # shuffled pairs. The gap is the signal; both rising together is not.
            "fb/F_B_cos_matched": _cos_rows(f, bn),
            "fb/F_B_cos_shuffled": _cos_rows(f, jnp.roll(bn, 1, axis=0)),
            # How far the online F has drifted from its EMA target. -> 0 means the target is
            # stale/redundant; large and growing means the bootstrap is chasing a moving object.
            "fb/F_target_drift_cos": _cos_rows(f, fb_),
            "fb/F_target_drift_rel": jnp.mean(jnp.linalg.norm(f - fb_, axis=-1))
            / (jnp.mean(jnp.linalg.norm(f, axis=-1)) + 1e-8),
            # Does the linear *_out projection preserve the encoder's geometry, or is it the
            # thing collapsing? Compare eff_rank of z_* against the projected rep.
            "fb/B_out_rank_ratio": _effective_rank(bn) / (_effective_rank(zb) + 1e-8),
            "fb/F_out_rank_ratio": _effective_rank(f) / (_effective_rank(zf) + 1e-8),
            "fb/M_sv_top1_frac": sv[0] / (jnp.sum(sv) + 1e-8),
            "fb/M_sv_top8_frac": jnp.sum(sv[:8]) / (jnp.sum(sv) + 1e-8),
            "fb/M_eff_rank": jnp.square(jnp.sum(sv)) / (jnp.sum(jnp.square(sv)) + 1e-8),
            # Fraction of the batch whose bootstrap was masked off. See TODO.md -- this currently
            # conflates genuine termination with time-limit truncation.
            "fb/terminal_frac": 1.0 - jnp.mean(not_terminal),
            "fb/batch_size": jnp.asarray(float(b)),
            "fb/d_over_b": jnp.asarray(float(self.rep_dim) / float(b)),
            **self._grad_balance(f, bn, fb_, bb_, not_terminal),
        }

    def _grad_balance(self, f_cur, b_next, f_boot, b_boot, not_terminal) -> dict[str, at.Array]:
        """Gradient-norm ratio of the two loss terms w.r.t. B -- this, not the loss magnitudes,
        is what says whether `fb_ortho_coeff` is mis-set. `l_ortho` is O(d) once B is orthonormal
        but O(d^2) while it is collapsed, so a large `fb_ortho_loss` early is the term doing its
        job.

        Cheap: two backward passes over a [b, d] array through the loss arithmetic only, with the
        result stop_gradded so the nested `jax.grad` cannot perturb the real graph.
        """
        gamma, quad_off = self.fb_gamma, self.fb_quadratic_offdiag_only

        def _fb_term(b):
            return fb_loss_terms(
                f_cur, b, f_boot, b_boot, gamma=gamma, not_terminal=not_terminal, quadratic_offdiag_only=quad_off
            )[0]

        def _ortho_term(b):
            return fb_loss_terms(
                f_cur, b, f_boot, b_boot, gamma=gamma, not_terminal=not_terminal, quadratic_offdiag_only=quad_off
            )[1]

        g_fb = jnp.linalg.norm(jax.grad(_fb_term)(b_next))
        g_or = jnp.linalg.norm(jax.grad(_ortho_term)(b_next)) * self.fb_ortho_coeff
        return {
            "fb/gradnorm_fb_wrt_B": jax.lax.stop_gradient(g_fb),
            "fb/gradnorm_ortho_wrt_B": jax.lax.stop_gradient(g_or),
            # >> 1 for a sustained stretch AFTER B_sigma2 has converged => ortho is swamping the
            # dynamics term and fb_ortho_coeff should come down. Large early is expected.
            "fb/gradratio_ortho_over_fb": jax.lax.stop_gradient(g_or / (g_fb + 1e-8)),
        }

    # ---------------------------------------------------------------------------------
    # Deployment
    # ---------------------------------------------------------------------------------

    def get_state_representations(self, observation, noisy_actions=None, timestep=None, z=None):
        """(B(s), None, kv_cache, prefix_mask, prefix_len). The guidance mixin reads element [0]
        to build z_r. The unused args exist for signature parity with Pi0RepBase."""
        del noisy_actions, timestep, z
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_out, _, kv_cache, prefix_mask, prefix_len = self._run_prefix(observation)
        rep, _, _ = self._rep("B", prefix_out, None, prefix_mask, None, deterministic=True)
        return rep, None, kv_cache, prefix_mask, prefix_len

    @override
    def grad_a_q(self, observation, actions, kv_cache, prefix_mask, prefix_len, z_r):
        """d/da [F(s,a)^T z_r] by autograd through the trained F encoder. z_r enters ONLY as the
        contraction target -- F carries no z-dependence in this model."""
        batch_size = actions.shape[0]
        z_r_b = jnp.broadcast_to(z_r, (batch_size, z_r.shape[-1]))
        timestep0 = jnp.zeros((batch_size,))  # F is only ever trained at clean actions (t=0)

        def scalar_q_and_aux(a):
            suffix_out, _, suffix_mask = self._suffix_forward(
                observation, a, timestep0, kv_cache, prefix_mask, prefix_len
            )
            # grad_scale_override=1.0 is load-bearing; see _rep's docstring.
            f_rep, _, _ = self._rep("F", None, suffix_out, None, suffix_mask, grad_scale_override=1.0)
            q = jnp.sum(f_rep * z_r_b, axis=-1)
            # Each example's Q depends only on its own action, so d(sum_i q_i)/da_i == d(q_i)/da_i.
            return jnp.sum(q), q

        return jax.grad(scalar_q_and_aux, has_aux=True)(actions)
