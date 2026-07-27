"""TD Successor-Features variant of pi05.

NOTE on naming: this model uses CRL's (pi05crl.py) phi/psi convention — psi = state readout
(L2-normalized state feature), phi = successor readout. This is INVERTED relative to
successor-feature literature (where phi is the state feature and psi the successor). The
convention is kept consistent with pi05crl.py to ease a future CRL+SF merge.

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


class _SFTarget(nnx.Variable):
    """Holds EMA target-network weights for the phi(s',a') bootstrap.

    A distinct (non-Param) Variable type so (a) the optimizer's trainable filter
    (All(Param, ...)) never updates it, and (b) the frozen-param bf16 cast (which matches on
    Param paths) leaves it float32 — EMA precision matters. It is updated in place by
    update_target_networks() (called by the train loop after each optimizer step, mirroring
    Pi0RepBase's psi_lagging_ema pattern), never inside the differentiated loss.
    """


class Pi0SF(Pi0RepBase):
    # phi(s,a) is the successor feature — its magnitude matters, so it stays unnormalized.
    # psi(s) is L2-normalized; the orthonormality loss on psi (see compute_loss) is the actual
    # anti-collapse mechanism — the norm constraint alone does not stop directional collapse.
    _normalize_phi = False
    _normalize_psi = True

    def __init__(self, config: pi0_config.Pi0SFConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.sf_loss_coeff = config.sf_loss_coeff
        self.sf_ortho_coeff = config.sf_ortho_coeff
        self.sf_gamma = config.sf_gamma
        self.sf_target_ema = config.sf_target_ema
        self.fb_train_goal_ratio = config.fb_train_goal_ratio
        # z conditions F via the action-expert AdaRMS pathway (keeps B=psi in the prefix z-free).
        # Note: with rep_backbone_grad_scale=0.0 the SF loss is cut off from the backbone, so
        # z_proj only receives gradient through the (grad-scaled) backbone path.
        self.z_proj = nnx.Linear(config.rep_dim, self.action_in_proj.out_features, rngs=rngs)
        # Optional EMA target network for the phi(s',a') bootstrap (phi-head pathway only; the
        # online backbone's next-half hidden states are reused, so no extra backbone forward).
        if self.sf_target_ema is not None:
            self._build_phi_target()

    @staticmethod
    def _relkey(relpath) -> str:
        """Encode a leaf's relative state path (a tuple) as a single string. Target leaves are
        stored in STRING-keyed dicts (not lists): a list container makes nnx emit integer path
        components, which State.to_pure_dict() -> flax.traverse_util.flatten_dict(sep="/") cannot
        join (int is not str) — that crashes weight-loading at init."""
        return "|".join(map(str, relpath))

    def _build_phi_target(self):
        """Create float32 _SFTarget copies of the phi-head pathway (phi_mix/phi_head/phi_proj),
        initialized equal to the online weights (no fresh init needed). The train loop EMAs them
        toward the online weights each step; the bootstrap reads them via _phi_target_rep."""
        self._phi_mix_target = _SFTarget(jnp.asarray(self.phi_mix.value, jnp.float32))
        head_flat = nnx.state(self.phi_head).flat_state()
        proj_flat = nnx.state(self.phi_proj).flat_state()
        self._phi_head_relpaths = list(head_flat.keys())
        self._phi_proj_relpaths = list(proj_flat.keys())
        # String-keyed dicts (see _relkey) so every state-path component is a str.
        self._phi_head_target = {
            self._relkey(p): _SFTarget(jnp.asarray(head_flat[p].value, jnp.float32)) for p in self._phi_head_relpaths
        }
        self._phi_proj_target = {
            self._relkey(p): _SFTarget(jnp.asarray(proj_flat[p].value, jnp.float32)) for p in self._phi_proj_relpaths
        }

    def _merge_target(self, online_module, targets, relpaths):
        """Rebuild `online_module` with target weights: swap the _SFTarget values into the live
        online leaves (via .replace, preserving each leaf's exact VariableState type/metadata —
        a fresh nnx.Param does not round-trip through the ToNNX bridge) and merge. `targets` is
        the string-keyed dict of _SFTarget leaves, looked up by _relkey(relpath)."""
        flat = nnx.state(online_module).flat_state()
        swapped = {p: flat[p].replace(targets[self._relkey(p)].value) for p in relpaths}
        return nnx.merge(nnx.graphdef(online_module), nnx.State.from_flat_path(swapped))

    def _phi_target_rep(self, prefix_hidden, suffix_hidden, prefix_mask, suffix_mask):
        """phi(s',a') from the EMA target head. Mirrors Pi0RepBase._compute_rep's phi branch but
        with target weights; reuses the (online) backbone hidden states passed in."""
        source = self.phi_input
        hidden, mask = (prefix_hidden, prefix_mask) if source == "state" else (suffix_hidden, suffix_mask)
        mem = self._mix_layers(hidden, self._phi_mix_target.value, self.rep_backbone_grad_scale)
        head = self._merge_target(self.phi_head, self._phi_head_target, self._phi_head_relpaths)
        proj = self._merge_target(self.phi_proj, self._phi_proj_target, self._phi_proj_relpaths)
        return proj(head(mem, mask, deterministic=True, dropout_rng=None)).astype(jnp.float32)

    def _blend_phi_target(self, decay: float):
        """In-place EMA of the phi target toward the online phi head: target <- decay*target +
        (1-decay)*online, over phi_mix/phi_head/phi_proj. decay=0.0 is a hard copy (sync). Mirrors
        Pi0RepBase._blend_lagging_psi; the .value mutation is what carries into nnx.state(model)
        in train_step. No-op container when sf_target_ema is None (targets are not built)."""
        self._phi_mix_target.value = decay * self._phi_mix_target.value + (1.0 - decay) * self.phi_mix.value.astype(
            jnp.float32
        )
        head_flat = nnx.state(self.phi_head).flat_state()
        for rp in self._phi_head_relpaths:
            t = self._phi_head_target[self._relkey(rp)]
            t.value = decay * t.value + (1.0 - decay) * head_flat[rp].value.astype(jnp.float32)
        proj_flat = nnx.state(self.phi_proj).flat_state()
        for rp in self._phi_proj_relpaths:
            t = self._phi_proj_target[self._relkey(rp)]
            t.value = decay * t.value + (1.0 - decay) * proj_flat[rp].value.astype(jnp.float32)

    @override
    def sync_target_networks(self):
        """Hard-copy the online phi head into the phi target (called once at init after the
        checkpoint merge, so a warm-started phi head is reflected in the target)."""
        super().sync_target_networks()  # base handles psi_lagging_ema (no-op for SF)
        if self.sf_target_ema is not None:
            self._blend_phi_target(0.0)

    @override
    def update_target_networks(self):
        """One EMA step of the phi target (called by train_step after each optimizer step)."""
        super().update_target_networks()  # base handles psi_lagging_ema (no-op for SF)
        if self.sf_target_ema is not None:
            self._blend_phi_target(self.sf_target_ema)

    @override
    def _adarms_cond(
        self, time_emb: at.Float[at.Array, "b emb"], z: at.Float[at.Array, "b d"] | None
    ) -> at.Float[at.Array, "b emb"]:
        return time_emb if z is None else time_emb + self.z_proj(z)

    def sample_mixed_z(self, z_goal, rng):
        """Mix goal-derived z (= B(future), detached) with random unit-sphere z.

        Returns (z, goal_frac) where with prob fb_train_goal_ratio z_i = z_goal_i, else random.
        """
        b, d = z_goal.shape
        rng_n, rng_m = jax.random.split(rng)
        z_rand = jax.random.normal(rng_n, (b, d))
        z_rand = z_rand / (jnp.linalg.norm(z_rand, axis=-1, keepdims=True) + 1e-6)
        use_goal = jax.random.uniform(rng_m, (b, 1)) < self.fb_train_goal_ratio
        z = jnp.where(use_goal, z_goal, z_rand)
        return z, jnp.mean(use_goal.astype(jnp.float32))

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        batch: dict[str, Any],
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """Semi-gradient SARSA TD successor-feature loss + flow-matching action loss.

        Three loss terms: (1) flow-matching action loss (standard pi05); (2) semi-gradient SARSA
        TD loss training phi(s,a) onto stop_gradient(psi) + gamma*phi(s',a'); (3) orthonormality
        loss E[psi psi^T] ~ I on psi. Because psi is stop-gradded in (2), term (3) is psi's only
        gradient source — this decoupling is what prevents the constant-feature collapse.

        Required batch keys: observation, future_observation, actions, next_observation,
        next_actions (SARSA a'). Optional: next_is_pad (terminal masking of the bootstrap).
        """
        _model.require_batch_keys(
            batch,
            ("observation", "future_observation", "actions", "next_observation", "next_actions"),
            type(self).__name__,
        )
        observation = batch["observation"]
        future_observation = batch["future_observation"]
        actions = batch["actions"]
        next_observation = batch["next_observation"]
        next_actions = batch["next_actions"]
        next_is_pad = batch.get("next_is_pad")

        preprocess_rng, next_preprocess_rng, noise_rng, time_rng, z_rng, dropout_rng = jax.random.split(rng, 6)
        # Independent head-dropout keys per rep (only used when train=True).
        phi_drop_rng, psi_drop_rng = jax.random.split(dropout_rng)
        deterministic = not train
        # Augment current/next independently (consistent with pi05crl) so phi(s,a) and the
        # bootstrap phi(s',a') cannot match on shared augmentation artifacts.
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, augment=self.image_augment
        )
        next_observation = _model.preprocess_observation(
            next_preprocess_rng, next_observation, train=train, augment=self.image_augment
        )

        # Task latent z (detached): B(future) for a fraction of the batch, random for the rest.
        # z = B(future): use the already-forwarded future_observation (no train-loop change).
        psi_goal = self.get_state_representations(future_observation)[0]
        if psi_goal is None:
            raise ValueError("Pi0SF's z = B(future) requires psi_input='state'")
        z_goal = jax.lax.stop_gradient(psi_goal)
        z, z_goal_frac = self.sample_mixed_z(z_goal, z_rng)
        z_stacked = jnp.concatenate([z, z], axis=0)  # same z for the [obs, next] stack halves

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

        # ---- (2) SF pass: clean actions (time=0 -> x_t == actions), stacked current+next ----
        obs_stacked = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0), observation, next_observation
        )
        actions_stacked = jnp.concatenate([actions, next_actions], axis=0)
        time0 = jnp.zeros((actions_stacked.shape[0],), dtype=jnp.float32)  # clean: x_t == actions
        _, prefix_hidden, suffix_hidden, prefix_mask, suffix_mask = self._forward(
            obs_stacked, actions_stacked, time0, z=z_stacked
        )

        # Reps via Pi0RepBase._compute_rep (input source per config; default psi(s) from the
        # prefix, phi(s,a) from the suffix). Layer axis is 0, so slice the batch on axis 1.
        def _rep_slice(which, sl, drop_rng, *, det):
            return self._compute_rep(
                which,
                prefix_hidden[:, sl],
                suffix_hidden[:, sl],
                prefix_mask[sl],
                suffix_mask[sl],
                deterministic=det,
                dropout_rng=drop_rng,
            )

        cur = slice(None, batch_size)
        nxt = slice(batch_size, None)
        # psi(s): CURRENT half, L2-normalized (anti-collapse).
        psi = self._finalize_rep("psi", _rep_slice("psi", cur, psi_drop_rng, det=deterministic))
        # phi(s,a): current half carries grad (and train-time dropout); the next half is the
        # stop-grad bootstrap and is computed deterministically — a noisy TD target would
        # inject variance into the regression.
        phi_sa = _rep_slice("phi", cur, phi_drop_rng, det=deterministic).astype(jnp.float32)
        # Bootstrap phi(s',a'): from the EMA target head when enabled (TD stability), else the
        # online head. Either way stop-gradded — it is the regression target, not a prediction.
        if self.sf_target_ema is not None:
            phi_next = self._phi_target_rep(
                prefix_hidden[:, nxt], suffix_hidden[:, nxt], prefix_mask[nxt], suffix_mask[nxt]
            )
        else:
            phi_next = _rep_slice("phi", nxt, None, det=True).astype(jnp.float32)
        phi_next = jax.lax.stop_gradient(phi_next)

        # Mask the bootstrap at episode boundaries (terminal): target = psi(s) only.
        if next_is_pad is None:
            not_terminal = jnp.ones((batch_size, 1), dtype=phi_next.dtype)
        else:
            not_terminal = (1.0 - next_is_pad.astype(phi_next.dtype))[:, None]
        # psi is stop-gradded in the target: the TD loss trains only phi (the successor), never
        # the feature it regresses onto. This is what removes the constant-psi/phi=c/(1-gamma)
        # collapse fixed point — psi's gradient comes solely from the orthonormality loss below.
        target = jax.lax.stop_gradient(psi) + self.sf_gamma * not_terminal * phi_next
        td_resid = phi_sa - target
        sf_loss = jnp.mean(jnp.square(td_resid))

        # ---- (3) Orthonormality loss on psi: E[psi psi^T] ~ I (FB's B-orthonormalization). ----
        # This is psi's ONLY gradient source (psi is stop-gradded in the TD target). Under CRL's
        # convention psi is unit-normalized, so its batch Gram is the pairwise cosine matrix; the
        # off-diagonal term decorrelates distinct states (kills directional collapse) and the diag
        # term (inert while psi is L2-normalized) would pin per-feature norm if it were not.
        gram = psi @ psi.T  # [b, b], gram[i, j] = cos(psi_i, psi_j) since psi is unit-norm
        # Off-diagonal reductions via a 0/1 mask, NOT boolean indexing: gram[~eye] returns a
        # value-dependent shape, which is illegal under jax.jit (the train loop jits this).
        off_mask = 1.0 - jnp.eye(batch_size, dtype=gram.dtype)  # 1 off-diagonal, 0 on-diagonal
        n_off = batch_size * (batch_size - 1)  # static (batch_size is a Python int)
        ortho_diag = -2.0 * jnp.mean(jnp.diagonal(gram))
        if n_off > 0:
            ortho_off = jnp.sum(jnp.square(gram) * off_mask) / n_off
            # Interpretable collapse metric: mean |cosine| over distinct-state pairs, in [0, 1]
            # (-> 1.0 as psi collapses to a single direction).
            psi_offdiag_cossim = jnp.sum(jnp.abs(gram) * off_mask) / n_off
        else:  # b == 1: no distinct-state pairs
            ortho_off = jnp.asarray(0.0)
            psi_offdiag_cossim = jnp.asarray(0.0)
        ortho_loss = self.sf_ortho_coeff * (ortho_diag + ortho_off)

        # Diagnostics (meaned by the train loop); layer-mix entropy watches for mix collapse.
        phi_mix_w = jax.nn.softmax(self.phi_mix.value.astype(jnp.float32))
        psi_mix_w = jax.nn.softmax(self.psi_mix.value.astype(jnp.float32))
        info = {
            "action_loss": jnp.mean(action_loss),
            "sf_loss": sf_loss,
            "ortho_loss": ortho_loss,
            "psi_offdiag_cossim": psi_offdiag_cossim,
            "psi_norm": jnp.mean(jnp.linalg.norm(psi, axis=-1)),
            "phi_norm": jnp.mean(jnp.linalg.norm(phi_sa, axis=-1)),
            # Across-batch mean/std of each rep (collapse -> std to 0). psi is L2-normalized,
            # so its std tracks directional spread; phi_sa is raw.
            **batch_rep_stats(psi, "psi"),
            **batch_rep_stats(phi_sa, "phi"),
            "sf_td_resid": jnp.mean(jnp.linalg.norm(td_resid, axis=-1)),
            "z_goal_frac": z_goal_frac,
            "phi_mix_entropy": -jnp.sum(phi_mix_w * jnp.log(phi_mix_w + 1e-6)),
            "psi_mix_entropy": -jnp.sum(psi_mix_w * jnp.log(psi_mix_w + 1e-6)),
            "phi_mix_max": jnp.max(phi_mix_w),
            "psi_mix_max": jnp.max(psi_mix_w),
        }
        total_loss = self.action_loss_coeff * action_loss + self.sf_loss_coeff * sf_loss + ortho_loss
        # "rep_loss" aliases sf_loss so the train loop's shared logging (which expects the CRL
        # model's key set) works unchanged.
        return total_loss, {"rep_loss": sf_loss, **info}
