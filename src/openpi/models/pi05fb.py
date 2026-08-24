"""Forward-Backward representations read off a frozen pi0.5 VLA.

The Touati & Ollivier FB objective applied to a SINGLE policy. FB approximates a policy's
successor measure by a rank-d bilinear form,

    M^pi(s, a, dx)  ~=  F(s, a)^T B(x) rho(dx),

and fits it by the L2(rho) Bellman residual. Both factors are trained by that loss, which is the
point of this file: `pi05sf.py` stop-gradients its feature map and leaves it with only an
orthonormality objective, which is provably indifferent to WHICH subspace the features span, so
its B is effectively a random whitened projection. FB's linear term `-E[F(s,a)^T B(s')]` is the
missing signal -- it pushes B(s') toward F(s,a) on observed transitions, so the loss chooses the
subspace from the dynamics.

Not zero-shot FB: a' comes from the dataset (SARSA on the frozen pi0.5) rather than from
argmax_a F(s,a,z)^T z, so the Bellman operator carries no z and, under E_rho[B B^T] = I, the
measure Bellman equation collapses to the successor-feature equation for feature map B,

    F(s,a) = E[B(s')] + gamma * E[F(s', a')].

F therefore takes no z argument: at deployment z_r = E_rho[r B] is CONTRACTED against F's output,
never fed into it, and F(s,a)^T z_r is the base policy's true Q for r projected onto span(B).
z-conditioning of F only becomes meaningful alongside a z-conditioned actor.

The heads are named F and B (not pi05sf.py's inverted phi/psi). The layer-mix / rep / target-EMA
plumbing is re-implemented under those names rather than inherited from `Pi0RepBase`, which
builds `phi_*`/`psi_*` trios unconditionally; the duplication leaves every existing config alone.
"""

import dataclasses
from typing import Any, Literal

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.models.rep_base import _MeanPoolRepHead
from openpi.models.rep_base import _RepHead
from openpi.models.rep_base import batch_rep_stats
from openpi.models.rep_grad_guidance import ValueGuidedSamplingMixin
from openpi.shared import array_typing as at


class _FBTarget(nnx.Variable):
    """EMA target weights for the F/B bootstrap.

    A distinct (non-Param) Variable type so (a) the optimizer's trainable filter All(Param, ...)
    never updates it and (b) the frozen-param bf16 cast, which matches Param paths, leaves it
    float32 -- EMA accumulation at bf16 loses the small (1-decay) increments entirely. Mutated in
    place by update_target_networks() after the optimizer step, never inside the loss.
    """


def _masked_mean(x, mask):
    """Mask-mean over the token axis. x: (b, t, d), mask: (b, t) -> (b, d)."""
    m = mask.astype(x.dtype)[..., None]
    return (x * m).sum(axis=1) / jnp.maximum(m.sum(axis=1), 1.0)


def fb_loss_terms(
    f_cur,
    b_next,
    f_target,
    b_target,
    *,
    gamma: float,
    not_terminal,
    quadratic_offdiag_only: bool = False,
):
    """The FB objective. A PURE function of arrays -- no model, no backbone, so the mathematics
    can be unit-tested against a tabular MDP with a known successor measure (see pi05fb_test.py).
    Every argument is [batch, rep_dim] except `not_terminal`, which is [batch].

    Writing M_ij = F(s_i,a_i)^T B(s'_j) and Mbar_ij = Fbar(s'_i,a'_i)^T Bbar(s'_j):

        L_fb    = 0.5 * mean_ij[ (M - gamma*(1-d_i)*Mbar)^2 ]  -  mean_i[ M_ii ]
        L_ortho = mean_{i != j}[ (B_i^T B_j)^2 ]  -  2 * mean_i[ ||B_i||^2 ]

    Three things are load-bearing:

    - The linear term is the diagonal only -- the `delta_{s'}` term of the measure Bellman
      equation, and the only thing preventing the trivial M == 0 solution.
    - L_ortho EXCLUDES the Gram diagonal, making it a U-statistic whose argmin is per-dimension
      variance 1 for any d/b. The plug-in form that includes i=j has argmin 1/(1 + d/b), i.e. 0.5
      whenever batch == rep_dim, which is what pinned `sf_psi_ortho_err` at ~0.50 in every
      previous run. Do not "simplify" it back.
    - The quadratic term INCLUDES the diagonal, matching the reference implementation; the j=i
      sample is an O(1/b) bias and `fb_quad_offdiag` reports the unbiased value alongside.

    The 0.5 / -1 scaling is the reference implementation's, so published `orth_coef` values carry.
    """
    batch_size = b_next.shape[0]
    rep_dim = b_next.shape[-1]
    eye = jnp.eye(batch_size, dtype=f_cur.dtype)
    off = 1.0 - eye
    n_off = jnp.maximum(batch_size * (batch_size - 1), 1)

    m_online = f_cur @ b_next.T
    m_boot = f_target @ b_target.T
    resid = m_online - gamma * not_terminal[:, None] * m_boot

    quad_all = jnp.mean(jnp.square(resid))
    quad_off = jnp.sum(jnp.square(resid) * off) / n_off
    quad = quad_off if quadratic_offdiag_only else quad_all
    diag = jnp.diagonal(m_online)
    lin = jnp.mean(diag)
    l_fb = 0.5 * quad - lin

    gram = b_next @ b_next.T
    l_ortho = jnp.sum(jnp.square(gram) * off) / n_off - 2.0 * jnp.mean(jnp.diagonal(gram))

    # --- diagnostics -------------------------------------------------------------------
    # The empirical E[B B^T] is the d x d covariance, NOT the b x b Gram used by the loss.
    cov = jax.lax.stop_gradient(b_next.T @ b_next / batch_size)
    ortho_err = jnp.sum(jnp.square(cov - jnp.eye(rep_dim, dtype=cov.dtype))) / rep_dim
    eigs = jnp.linalg.eigvalsh(cov)
    # Judge orthonormality by B_sigma2, not B_ortho_err: the latter carries a finite-sample noise
    # floor of ~(d+1)/b even at a population covariance of exactly I (1.03 at d=b=32, 0.26 at
    # d=32/b=128), so it is not comparable across rep_dim/batch settings. B_sigma2 is the
    # scale-free version and must go to 1.0.
    b_sigma2 = jnp.mean(jnp.diagonal(cov))

    # Ranking of the true next state against the b-1 distractors. This is the single best
    # "is it learning" signal: 1/b at random init, -> 1 as the factorization localizes where the
    # policy actually goes. A loss that falls while this stays at chance means F and B found a
    # degenerate way to shrink the residual.
    m_det = jax.lax.stop_gradient(m_online)
    better = jnp.sum((m_det > jnp.diagonal(m_det)[:, None]).astype(jnp.float32) * off, axis=-1)
    rank_top1 = jnp.mean((better == 0).astype(jnp.float32))
    rank_mrr = jnp.mean(1.0 / (better + 1.0))

    offdiag_mean = jnp.sum(m_det * off) / n_off
    diag_mean = jnp.mean(jnp.diagonal(m_det))

    metrics = {
        "fb_quad": quad,
        "fb_quad_offdiag": quad_off,
        "fb_lin": lin,
        "fb_td_resid_rel": jnp.mean(jnp.square(resid)) / (jnp.mean(jnp.square(m_online)) + 1e-8),
        "M_rank_top1": rank_top1,
        "M_rank_mrr": rank_mrr,
        "M_diag_mean": diag_mean,
        "M_offdiag_mean": offdiag_mean,
        "M_contrast": (diag_mean - offdiag_mean) / (jnp.std(m_det) + 1e-8),
        "B_sigma2": b_sigma2,
        "B_ortho_err": ortho_err,
        "B_cov_eig_min": jnp.min(eigs),
        "B_cov_eig_max": jnp.max(eigs),
    }
    return l_fb, l_ortho, metrics


class Pi0FB(ValueGuidedSamplingMixin, _pi0.Pi0):
    """Forward-Backward representation heads on a frozen pi0.5 backbone. See module docstring."""

    def __init__(self, config: "pi0_config.Pi0FBConfig", rngs: nnx.Rngs):
        super().__init__(config, rngs)
        paligemma_config = _gemma.get_config(
            config.paligemma_variant,
            lora_rank=config.paligemma_lora_rank,
            lora_alpha=config.paligemma_lora_alpha,
        )
        action_expert_config = _gemma.get_config(
            config.action_expert_variant,
            lora_rank=config.action_expert_lora_rank,
            lora_alpha=config.action_expert_lora_alpha,
        )

        self.F_input = config.F_input
        self.B_input = config.B_input
        # Plain attributes, not properties: nnx's __setattr__ rejects a subclass shadowing
        # Pi0.__init__'s phi_input/psi_input with a read-only property. Overwriting them is also
        # what lets ValueGuidedSamplingMixin work unmodified -- B is this model's psi.
        self.psi_input = config.B_input
        self.phi_input = config.F_input
        self.rep_dim = config.rep_dim
        self.rep_backbone_grad_scale = config.rep_backbone_grad_scale
        self.rep_head_dropout = config.rep_head_dropout
        self.include_proprio = config.include_proprio
        self.F_include_prefix = config.F_include_prefix
        self.fb_gamma = config.fb_gamma
        self.fb_ortho_coeff = config.fb_ortho_coeff
        self.fb_target_ema = config.fb_target_ema
        self.fb_quadratic_offdiag_only = config.fb_quadratic_offdiag_only
        self.guidance_schedule = config.guidance_schedule
        self.guidance_grad_clip_norm = config.guidance_grad_clip_norm
        self.guidance_eval_at_xt = config.guidance_eval_at_xt
        self.normalize_z_r_default = config.normalize_z_r

        widths = {"state": paligemma_config.width, "state_action": action_expert_config.width}
        b_mem_width = widths[config.B_input]
        f_mem_width = widths[config.F_input]

        def _head_config(width):
            return dataclasses.replace(paligemma_config, width=width, depth=config.rep_head_depth, lora_configs={})

        def _build_head(mem_width: int):
            if config.rep_head_kind == "meanpool":
                return _MeanPoolRepHead(config.rep_head_depth, config.dtype, dropout=config.rep_head_dropout)
            if config.rep_head_kind != "attn":
                raise ValueError(f"unknown rep_head_kind={config.rep_head_kind!r}; expected 'attn' or 'meanpool'")
            return _RepHead(_head_config(mem_width), config.dtype, dropout=config.rep_head_dropout)

        b_head = nnx_bridge.ToNNX(_build_head(b_mem_width))
        b_head.lazy_init(jnp.zeros((1, 2, b_mem_width)), jnp.ones((1, 2), dtype=bool), rngs=rngs)
        self.B_head = b_head
        f_head = nnx_bridge.ToNNX(_build_head(f_mem_width))
        f_head.lazy_init(jnp.zeros((1, 2, f_mem_width)), jnp.ones((1, 2), dtype=bool), rngs=rngs)
        self.F_head = f_head

        # Learned softmax mix over all backbone layers, initialized to favor early layers.
        # `_scale_grad` cuts the backbone's backward pass but NOT these logits, so the mixes
        # train even under a fully frozen backbone -- the heads are read-only probes over a
        # learned layer mixture.
        num_backbone_layers = paligemma_config.depth
        self.B_mix = nnx.Param(jnp.linspace(1.0, -1.0, num_backbone_layers))
        self.F_mix = nnx.Param(jnp.linspace(1.0, -1.0, num_backbone_layers))

        # obs.state is zero-padded to action_dim by PadStatesAndActions, so this is the padded
        # width (8 real LIBERO dims + 24 structural zeros the projection learns to ignore).
        proprio_dim = config.action_dim if config.include_proprio else 0
        prefix_extra = paligemma_config.width if config.F_include_prefix else 0
        self.B_proj = nnx.Linear(b_mem_width + proprio_dim, config.rep_dim, rngs=rngs)
        self.F_proj = nnx.Linear(f_mem_width + proprio_dim + prefix_extra, config.rep_dim, rngs=rngs)

        if self.fb_target_ema is not None:
            self._build_fb_targets()

    # ---------------------------------------------------------------------------------
    # Representation plumbing (F/B-named counterparts of Pi0RepBase's phi/psi machinery)
    # ---------------------------------------------------------------------------------

    @staticmethod
    def _scale_grad(x, scale: float):
        # 0 => full stop-gradient (cleanest graph); 1 => identity; else partial.
        if scale == 0.0:
            return jax.lax.stop_gradient(x)
        if scale == 1.0:
            return x
        return scale * x + (1.0 - scale) * jax.lax.stop_gradient(x)

    def _mix(self, hidden, mix_logits, scale: float):
        # hidden: (L, b, t, d); mix_logits: (L,). Returns (b, t, d) float32.
        hidden = self._scale_grad(hidden, scale)
        weights = jax.nn.softmax(mix_logits.astype(jnp.float32))
        return jnp.einsum("l,lbtd->btd", weights, hidden.astype(jnp.float32))

    def _assemble(self, which: Literal["F", "B"], pooled, prefix_mem, state, proj):
        """Concatenate the optional extra inputs onto the pooled vector, then project."""
        parts = [pooled.astype(jnp.float32)]
        if which == "F" and self.F_include_prefix:
            parts.append(prefix_mem.astype(jnp.float32))
        if self.include_proprio:
            if state is None:
                raise ValueError("include_proprio=True but no state was supplied")
            parts.append(state.astype(jnp.float32))
        x = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)
        return proj(x).astype(jnp.float32)

    def _rep(
        self,
        which: Literal["F", "B"],
        prefix_hidden,
        suffix_hidden,
        prefix_mask,
        suffix_mask,
        state,
        *,
        deterministic: bool = True,
        dropout_rng=None,
        grad_scale_override: float | None = None,
    ):
        source = self.F_input if which == "F" else self.B_input
        hidden, mask = (prefix_hidden, prefix_mask) if source == "state" else (suffix_hidden, suffix_mask)
        if hidden is None:
            seq = "prefix" if source == "state" else "suffix"
            raise ValueError(f"{which}_input={source!r} requires {seq} hidden states")
        grad_scale = self.rep_backbone_grad_scale if grad_scale_override is None else grad_scale_override
        mix = self.F_mix if which == "F" else self.B_mix
        head = self.F_head if which == "F" else self.B_head
        proj = self.F_proj if which == "F" else self.B_proj
        pooled = head(
            self._mix(hidden, mix.value, grad_scale), mask, deterministic=deterministic, dropout_rng=dropout_rng
        )
        prefix_mem = None
        if which == "F" and self.F_include_prefix:
            prefix_mem = _masked_mean(self._mix(prefix_hidden, mix.value, grad_scale), prefix_mask)
        return self._assemble(which, pooled, prefix_mem, state, proj)

    # ---------------------------------------------------------------------------------
    # EMA target networks (same pattern as pi05sf.py, generalized over the two trios)
    # ---------------------------------------------------------------------------------

    @staticmethod
    def _relkey(relpath) -> str:
        """Encode a leaf's relative state path (a tuple) as a single string. Target leaves live in
        STRING-keyed dicts, not lists: a list container makes nnx emit integer path components,
        which State.to_pure_dict() -> flatten_dict(sep="/") cannot join, crashing weight loading
        at init."""
        return "|".join(map(str, relpath))

    def _snapshot(self, module):
        flat = nnx.state(module).flat_state()
        relpaths = list(flat.keys())
        return {self._relkey(p): _FBTarget(jnp.asarray(flat[p].value, jnp.float32)) for p in relpaths}, relpaths

    def _build_fb_targets(self):
        self._F_mix_target = _FBTarget(jnp.asarray(self.F_mix.value, jnp.float32))
        self._B_mix_target = _FBTarget(jnp.asarray(self.B_mix.value, jnp.float32))
        self._F_head_target, self._F_head_relpaths = self._snapshot(self.F_head)
        self._F_proj_target, self._F_proj_relpaths = self._snapshot(self.F_proj)
        self._B_head_target, self._B_head_relpaths = self._snapshot(self.B_head)
        self._B_proj_target, self._B_proj_relpaths = self._snapshot(self.B_proj)

    def _merge_target(self, online_module, targets, relpaths):
        """Rebuild `online_module` with target weights, preserving each leaf's exact VariableState
        type via .replace (a fresh nnx.Param does not round-trip through the ToNNX bridge)."""
        flat = nnx.state(online_module).flat_state()
        swapped = {p: flat[p].replace(targets[self._relkey(p)].value) for p in relpaths}
        return nnx.merge(nnx.graphdef(online_module), nnx.State.from_flat_path(swapped))

    def _target_rep(self, which: Literal["F", "B"], prefix_hidden, suffix_hidden, prefix_mask, suffix_mask, state):
        """F-bar / B-bar from the EMA target weights, reusing the ONLINE backbone hidden states
        passed in (no extra backbone forward)."""
        source = self.F_input if which == "F" else self.B_input
        hidden, mask = (prefix_hidden, prefix_mask) if source == "state" else (suffix_hidden, suffix_mask)
        mix_t = self._F_mix_target if which == "F" else self._B_mix_target
        head = self._merge_target(
            getattr(self, f"{which}_head"),
            getattr(self, f"_{which}_head_target"),
            getattr(self, f"_{which}_head_relpaths"),
        )
        proj = self._merge_target(
            getattr(self, f"{which}_proj"),
            getattr(self, f"_{which}_proj_target"),
            getattr(self, f"_{which}_proj_relpaths"),
        )
        mem = self._mix(hidden, mix_t.value, self.rep_backbone_grad_scale)
        pooled = head(mem, mask, deterministic=True, dropout_rng=None)
        prefix_mem = None
        if which == "F" and self.F_include_prefix:
            prefix_mem = _masked_mean(self._mix(prefix_hidden, mix_t.value, self.rep_backbone_grad_scale), prefix_mask)
        return self._assemble(which, pooled, prefix_mem, state, proj)

    def _blend(self, decay: float):
        """In-place EMA of both target trios toward their online counterparts. decay=0.0 is a
        hard copy. The .value mutation is what carries into nnx.state(model) in train_step."""
        if self.fb_target_ema is None:
            return
        for which in ("F", "B"):
            mix_t = getattr(self, f"_{which}_mix_target")
            mix_t.value = decay * mix_t.value + (1.0 - decay) * getattr(self, f"{which}_mix").value.astype(jnp.float32)
            for kind in ("head", "proj"):
                online_flat = nnx.state(getattr(self, f"{which}_{kind}")).flat_state()
                targets = getattr(self, f"_{which}_{kind}_target")
                for rp in getattr(self, f"_{which}_{kind}_relpaths"):
                    t = targets[self._relkey(rp)]
                    t.value = decay * t.value + (1.0 - decay) * online_flat[rp].value.astype(jnp.float32)

    def sync_target_networks(self):
        """Called once by `scripts/train.py` after the checkpoint merge."""
        self._blend(0.0)

    def update_target_networks(self):
        """Called every step by `scripts/train.py` after the optimizer update."""
        self._blend(self.fb_target_ema if self.fb_target_ema is not None else 0.0)

    # ---------------------------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------------------------

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        batch: dict[str, Any],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """The FB loss. No flow-matching action loss: this model never trains the policy, and
        computing a BC loss it would multiply by zero would cost a whole extra forward pass.

        Required batch keys: observation, actions, next_observation, next_actions.
        Optional: next_is_pad (terminal masking of the bootstrap).
        """
        _model.require_batch_keys(
            batch, ("observation", "actions", "next_observation", "next_actions"), type(self).__name__
        )
        actions = batch["actions"]
        next_actions = batch["next_actions"]
        next_is_pad = batch.get("next_is_pad")

        pre_rng, next_pre_rng, drop_rng = jax.random.split(rng, 3)
        f_drop_rng, b_drop_rng = jax.random.split(drop_rng)
        deterministic = not train
        # Augment current/next independently so F(s,a) and the bootstrap cannot agree on shared
        # augmentation artifacts.
        observation = _model.preprocess_observation(pre_rng, batch["observation"], train=train)
        next_observation = _model.preprocess_observation(next_pre_rng, batch["next_observation"], train=train)

        batch_size = actions.shape[0]
        obs_stacked = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), observation, next_observation
        )
        actions_stacked = jnp.concatenate([actions, next_actions], axis=0)
        time0 = jnp.zeros((actions_stacked.shape[0],), dtype=jnp.float32)  # clean actions: x_t == a
        # No z=: F is not z-conditioned (see module docstring), so the frozen AdaRMS z pathway --
        # whose z_proj can never train under a stopped-gradient backbone -- is left unused.
        _, prefix_hidden, suffix_hidden, prefix_mask, suffix_mask = self._forward(obs_stacked, actions_stacked, time0)

        cur = slice(None, batch_size)
        nxt = slice(batch_size, None)
        state_cur = observation.state
        state_nxt = next_observation.state

        def _slice(which, sl, state, *, det, rng_):
            return self._rep(
                which,
                prefix_hidden[:, sl],
                suffix_hidden[:, sl],
                prefix_mask[sl],
                suffix_mask[sl],
                state,
                deterministic=det,
                dropout_rng=rng_,
            )

        # F(s,a) from the current half's suffix; B(s') from the next half's prefix. The b next
        # states double as the x ~ rho sample the quadratic term integrates over, which is what
        # makes M a b x b matrix rather than requiring a separate draw.
        f_cur = _slice("F", cur, state_cur, det=deterministic, rng_=f_drop_rng)
        b_next = _slice("B", nxt, state_nxt, det=deterministic, rng_=b_drop_rng)

        if self.fb_target_ema is not None:
            f_boot = self._target_rep(
                "F", prefix_hidden[:, nxt], suffix_hidden[:, nxt], prefix_mask[nxt], suffix_mask[nxt], state_nxt
            )
            b_boot = self._target_rep(
                "B", prefix_hidden[:, nxt], suffix_hidden[:, nxt], prefix_mask[nxt], suffix_mask[nxt], state_nxt
            )
        else:
            f_boot = _slice("F", nxt, state_nxt, det=True, rng_=None)
            b_boot = _slice("B", nxt, state_nxt, det=True, rng_=None)
        f_boot = jax.lax.stop_gradient(f_boot)
        b_boot = jax.lax.stop_gradient(b_boot)

        if next_is_pad is None:
            not_terminal = jnp.ones((batch_size,), dtype=f_cur.dtype)
        else:
            not_terminal = 1.0 - next_is_pad.astype(f_cur.dtype)

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
            # Alias so shared logging that expects the CRL/SF key set keeps working.
            "rep_loss": l_fb,
            **metrics,
            **batch_rep_stats(f_cur, "F"),
            **batch_rep_stats(b_next, "B"),
            "F_norm": jnp.mean(jnp.linalg.norm(f_cur, axis=-1)),
            "B_norm": jnp.mean(jnp.linalg.norm(b_next, axis=-1)),
            "F_target_norm": jnp.mean(jnp.linalg.norm(f_boot, axis=-1)),
            "F_mix_entropy": _entropy(self.F_mix.value),
            "B_mix_entropy": _entropy(self.B_mix.value),
            "F_mix_max": jnp.max(jax.nn.softmax(self.F_mix.value.astype(jnp.float32))),
            "B_mix_max": jnp.max(jax.nn.softmax(self.B_mix.value.astype(jnp.float32))),
        }
        # train.py means this; broadcasting keeps the (*b, ah) shape the BaseModel contract
        # declares without inventing a per-timestep decomposition the FB loss does not have.
        chunked = jnp.broadcast_to(total, (batch_size, self.action_horizon))
        return chunked, info

    # ---------------------------------------------------------------------------------
    # Deployment: ValueGuidedSamplingMixin interface
    # ---------------------------------------------------------------------------------

    def get_state_representations(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions | None = None,
        timestep: at.Float[at.Array, " b"] | None = None,
        z: at.Float[at.Array, "b d"] | None = None,
    ):
        """(B(s), None, kv_cache, prefix_mask, prefix_len) from one prefix pass. The mixin reads
        element [0] to build z_r.

        `noisy_actions`/`timestep`/`z` exist only for signature parity with
        Pi0RepBase.get_state_representations and are unused: B needs no action, and F takes no z.
        """
        del noisy_actions, timestep, z
        if self.B_input != "state":
            raise ValueError(f"B must read the prefix to be computable without actions; got B_input={self.B_input!r}")
        observation = _model.preprocess_observation(None, observation, train=False)
        _, prefix_hidden, kv_cache, prefix_mask, prefix_len = self._run_prefix(observation)
        b_rep = self._rep("B", prefix_hidden, None, prefix_mask, None, observation.state, deterministic=True)
        return b_rep, None, kv_cache, prefix_mask, prefix_len

    @override
    def grad_a_q(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_len: int,
        z_r: at.Float[at.Array, "d"],
    ) -> tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, "b"]]:
        """d/da [F(s,a)^T z_r] by autograd through the trained F network.

        z_r enters ONLY as the contraction target -- it is never fed into F, because F carries no
        z-dependence in this model (module docstring). Costs one backward pass through the action
        expert per denoising step; the PaliGemma prefix is cached and never revisited.
        """
        if self.F_include_prefix:
            raise NotImplementedError(
                "F_include_prefix=True is training-only for now: guided sampling reaches F through "
                "_suffix_forward, which returns no prefix hidden states (they live in kv_cache). "
                "Supporting it needs the pooled prefix threaded through sample_actions_guided."
            )
        batch_size = actions.shape[0]
        z_r_batched = jnp.broadcast_to(z_r, (batch_size, z_r.shape[-1]))
        timestep0 = jnp.zeros((batch_size,))  # F is only ever trained at clean actions (t=0)

        def scalar_q_and_aux(a):
            # grad_scale_override=1.0 is load-bearing: _rep otherwise stop_gradients
            # suffix_hidden, the ONLY path from `a` to F, so under a frozen backbone this
            # jax.grad returns exactly zeros. That stop_gradient is a training-time optimization;
            # the inference-time Jacobian is what guidance is defined as.
            _, suffix_hidden, suffix_mask = self._suffix_forward(
                observation, a, timestep0, kv_cache, prefix_mask, prefix_len
            )
            f_rep = self._rep("F", None, suffix_hidden, None, suffix_mask, observation.state, grad_scale_override=1.0)
            q_per_example = jnp.sum(f_rep * z_r_batched, axis=-1)  # [b]
            # Each example's Q depends only on its own action, so d(sum_i q_i)/da_i == d(q_i)/da_i.
            return jnp.sum(q_per_example), q_per_example

        return jax.grad(scalar_q_and_aux, has_aux=True)(actions)


def _entropy(mix_logits) -> at.Float[at.Array, ""]:
    """Entropy of the layer-mixing softmax. Watches for mix collapse onto a single layer."""
    w = jax.nn.softmax(mix_logits.astype(jnp.float32))
    return -jnp.sum(w * jnp.log(w + 1e-8))
