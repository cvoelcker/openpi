"""Shared scaffolding for value-guided denoising on top of Pi0SF-style successor features.

The deployment recipe for "steer the base VLA policy toward high-Q actions at inference time,
given only a handful of reward-labeled samples for a new task":

  1. `test_time_adapt`: FB's zero-shot reward embedding z_r = E[r(s) * psi(s)]. A single weighted
     average, no gradient steps, and deliberately the same computation regardless of how
     grad_a Q is estimated downstream.
  2. `grad_a_q`: grad_a Q_{z_r}(s, a), left abstract for subclasses to provide.
  3. `sample_actions_guided`: the denoising loop. The BASE flow velocity is used exactly as
     trained -- the action loss never conditions on z -- and z_r enters only through an added
     value-gradient term evaluated at the one-step clean-action estimate:

         x_hat_0(x_t, t) = x_t - t * v_bc(x_t, t)
         v_guided(x_t, t) = v_bc(x_t, t) - weight(t) * clip(grad_a Q_{z_r}(s, x_hat_0(x_t, t)))

     x_hat_0 is exact algebra on the rectified-flow interpolant x_t = t*noise + (1-t)*actions,
     not an approximation -- the same trick as Tweedie plug-ins in guided diffusion.

Deployment-time only: neither training nor the target-network machinery is touched.
"""

import dataclasses
import logging
from typing import Literal

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# Sentinel distinguishing "caller didn't pass guidance_grad_clip_norm" (fall back to the model's
# configured default) from "caller explicitly passed None" (disable clipping for this call).
_UNSET = object()


@dataclasses.dataclass(frozen=True)
class RewardSamples:
    """A handful of reward-labeled states for few-shot test-time adaptation.

    `observation` is a batch of K states (K = a few-shot sample count -- a handful of scripted
    or human-labeled examples for the new task, NOT a training-sized batch or a full rollout
    dataset). `reward` is the scalar reward r(s) attached to each one. FB's zero-shot recipe
    only needs enough samples to Monte-Carlo estimate an expectation.
    """

    observation: _model.Observation
    reward: at.Float[at.Array, "k"]


@dataclasses.dataclass(frozen=True)
class GuidanceWeightSchedule:
    """Guidance strength as a function of the flow-matching ODE's LOCAL time `t` (in [0, 1],
    1=pure noise, 0=clean action) -- NOT a training-step schedule (see pi0_config.CoeffSchedule
    for that). Default is `"constant"`, i.e. no behavior change vs. a fixed scalar weight.

    `"linear_anneal"`/`"power_anneal"` fade guidance in as t -> 0: `x_hat_0` is a one-step
    extrapolation from the current noisy iterate, so it is least reliable at large t (far from
    a clean action) -- annealing guidance down at large t avoids steering off a poor estimate.
    """

    base_weight: float = 1.0
    shape: Literal["constant", "linear_anneal", "power_anneal"] = "constant"
    power: float = 1.0  # only used when shape == "power_anneal"

    def __call__(self, t: at.Float[at.Array, ""]) -> at.Float[at.Array, ""]:
        if self.shape == "constant":
            return jnp.asarray(self.base_weight, dtype=jnp.float32)
        frac = jnp.clip(1.0 - t, 0.0, 1.0)
        if self.shape == "linear_anneal":
            return self.base_weight * frac
        if self.shape == "power_anneal":
            return self.base_weight * jnp.power(frac, self.power)
        raise ValueError(f"unknown GuidanceWeightSchedule shape {self.shape!r}")


def _cos_sim(pred, target, eps: float = 1e-8) -> at.Float[at.Array, ""]:
    """Mean cosine similarity between two batches of same-shaped arrays -- per DEBUG.md's central
    lesson, a raw (unbounded) loss/norm value alone is a poor quality signal; this gives a
    bounded, directly-interpretable [-1, 1] diagnostic instead. `target` is stop-gradded (the
    original use case: comparing a learned head's prediction against an exact ground-truth
    target); harmless here too, since none of this module's `return_diagnostics` code path is
    ever differentiated through by a caller."""
    p = pred.reshape(pred.shape[0], -1).astype(jnp.float32)
    t = jax.lax.stop_gradient(target.reshape(target.shape[0], -1).astype(jnp.float32))
    num = jnp.sum(p * t, axis=-1)
    den = jnp.linalg.norm(p, axis=-1) * jnp.linalg.norm(t, axis=-1) + eps
    return jnp.mean(num / den)


def _clip_guidance_grad(
    grad: at.Float[at.Array, "b ah ad"], clip_norm: float | None
) -> tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, "b"]]:
    """Per-example norm clip on the guidance gradient before it perturbs the base velocity.

    Distinct from the training-side global `optax.clip_by_global_norm` (which only ever bounds
    gradients w.r.t. trainable PARAMETERS during training) -- this guards the DEPLOYMENT-time
    guidance term, which can blow up if grad_a_q's estimate is poorly calibrated (e.g. early in
    training, or from an imperfect distillation head), pushing the guided ODE trajectory off
    the action manifold the base policy was actually trained on. Returns (clipped_grad,
    per-example grad_norm) -- the norm is returned unclipped, for diagnostics.
    """
    batch_size = grad.shape[0]
    grad_flat = grad.reshape(batch_size, -1)
    grad_norm = jnp.linalg.norm(grad_flat, axis=-1)
    if clip_norm is None:
        return grad, grad_norm
    scale = jnp.minimum(1.0, clip_norm / jnp.maximum(grad_norm, 1e-8))
    return grad * scale[:, None, None], grad_norm


class ValueGuidedSamplingMixin:
    """Adds few-shot test-time adaptation and value-guided denoising on top of a Pi0SF-style
    model. Mix in ALONGSIDE Pi0SF (or a subclass): `class Foo(ValueGuidedSamplingMixin, Pi0SF)`.
    Defines no state of its own (no __init__) -- safe to combine with nnx.Module subclasses.

    Subclasses must implement `grad_a_q`, and may set instance attributes `guidance_schedule`
    (a `GuidanceWeightSchedule`), `guidance_grad_clip_norm` (float | None), and
    `normalize_z_r_default` (bool) in their own `__init__` to configure the defaults used below.
    """

    def test_time_adapt(
        self, reward_samples: RewardSamples, *, normalize_z_r: bool | None = None
    ) -> at.Float[at.Array, "d"]:
        """FB's zero-shot reward embedding z_r = E[r(s) * psi(s)], a single weighted average over
        psi(s) -- no gradient steps, no rollout. Requires psi_input="state".

        Uses z* = z_r directly (vanilla FB's self-consistency choice) rather than a Langevin-
        refined z*.

        `normalize_z_r`: z_r averages unit-norm psi(s) vectors, so disagreeing samples can leave
        its norm well below 1 -- off distribution for an AdaRMS pathway only ever trained on
        unit-norm z. True rescales it back onto the unit sphere. Falls back to
        `self.normalize_z_r_default` when not passed.
        """
        psi_k = self.get_state_representations(reward_samples.observation)[0]
        if psi_k is None:
            raise ValueError(
                "test_time_adapt requires psi_input='state' (z_r = E[r * psi(s)] needs the "
                "state-only readout); got psi_input="
                f"{getattr(self, 'psi_input', None)!r}."
            )
        z_r = jnp.mean(reward_samples.reward[:, None] * psi_k.astype(jnp.float32), axis=0)
        should_normalize = normalize_z_r if normalize_z_r is not None else getattr(self, "normalize_z_r_default", False)
        if should_normalize:
            z_r = z_r / (jnp.linalg.norm(z_r) + 1e-6)
        return z_r

    def grad_a_q(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
        kv_cache,
        prefix_mask: at.Bool[at.Array, "b s"],
        prefix_len: int,
        z_r: at.Float[at.Array, "d"],
    ) -> tuple[at.Float[at.Array, "b ah ad"], at.Float[at.Array, "b"] | None]:
        """Returns (grad_a Q_{z_r}(s,a), Q_{z_r}(s,a) or None). The scalar Q value is exposed for
        deployment-time diagnostics (see sample_actions_guided's diagnostics dict below) --
        `None` is only a valid return for a hypothetical subclass whose gradient estimator can't
        cheaply produce an accompanying value; both current subclasses (Pi0SFAutogradGuidance,
        Pi0SFLearnedJacobian) return a real value, since it's a free byproduct in both cases."""
        raise NotImplementedError

    def sample_actions_guided(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        z_r: at.Float[at.Array, "d"],
        *,
        num_steps: int = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        guidance_schedule: GuidanceWeightSchedule | None = None,
        guidance_grad_clip_norm=_UNSET,
        guidance_eval_at_xt=_UNSET,
        bc_velocity_noise: float = 0.0,
        bc_velocity_bias: float = 0.0,
        return_diagnostics: bool = False,
    ):
        """Denoising-action guidance: base (unconditioned) BC velocity plus a value-gradient
        term steering toward high Q_{z_r} regions, evaluated at each step's one-step clean-action
        estimate (see module docstring). Mirrors Pi0.sample_actions' ODE-integration structure.

        `guidance_schedule` and `guidance_grad_clip_norm` fall back to the instance defaults; pass
        `guidance_grad_clip_norm=None` explicitly to disable clipping for a single call.

        `bc_velocity_noise` and `bc_velocity_bias` are the positive control: degrade a near-perfect
        policy, then ask whether guidance steers it back. Both perturb `v_bc` rather than the final
        action, so they sit at the same point in the computation as guidance and can therefore be
        cancelled by it -- perturbing downstream would guarantee a null by construction. Both are
        multiplied in rather than branched on, so 0.0 is numerically identical to the unperturbed
        path, and neither stream depends on the guidance weight, so a scale-0 and a scale-g arm see
        identical perturbations.

        The two differ in what they measure. Independent per-step noise does not degrade this
        policy at all (98.8% success at sigma=1.2, where the per-element perturbation exceeds the
        per-element velocity) because 10 denoising steps average it out. A FIXED offset does not
        average out: it accumulates across steps and biases every replan the same way, which is
        both a correctable error and the same character of error as guidance itself.

        The measurement `bc_velocity_bias` exists for is `bc_bias_pushback_cos`, not success rate:
        guidance contributes `-weight*grad_clipped` while the injected error is `+bias`, so >0 means
        guidance opposes the bias (Q knows about the error), ~0 indifferent, <0 amplifying. Being
        per-step and per-batch-element, it has far more samples than an episode-level success rate
        and is not bounded by the ~14.5% replicate-to-replicate discordance.
        """
        schedule = (
            guidance_schedule
            if guidance_schedule is not None
            else getattr(self, "guidance_schedule", GuidanceWeightSchedule())
        )
        clip_norm = (
            guidance_grad_clip_norm
            if guidance_grad_clip_norm is not _UNSET
            else getattr(self, "guidance_grad_clip_norm", 10.0)
        )
        # Same _UNSET fallback convention as clip_norm: the instance attribute unless this call
        # site overrides it explicitly.
        eval_at_xt = (
            getattr(self, "guidance_eval_at_xt", False) if guidance_eval_at_xt is _UNSET else bool(guidance_eval_at_xt)
        )

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        # Folded off `rng` so the draw above is unchanged; see the docstring.
        bc_noise_rng = jax.random.fold_in(rng, 0x5EED)
        # Hardcoded key, not folded off `rng`, so the direction is byte-identical across steps,
        # episodes, tasks and guidance arms -- only its magnitude is a knob.
        bias_dir = jax.random.normal(jax.random.key(0xB1A5), (self.action_horizon, self.action_dim))
        # Unit RMS per element, matching the `bc_velocity_noise` convention, so the two knobs are
        # directly comparable. Whole-block unit norm instead made a bias of 0.4 a 2.2% perturbation
        # against ||v_bc|| ~ 18.3 -- ~50x too small, and every swept value returned the baseline.
        bias_dir = bias_dir / (jnp.sqrt(jnp.mean(jnp.square(bias_dir))) + 1e-8)
        bias_vec = jnp.asarray(bc_velocity_bias, jnp.float32) * bias_dir

        # One prefix pass, cached and reused at every denoising step (both for the base velocity
        # and for grad_a_q's own suffix pass(es)) -- neither depends on z, so this is shared.
        _, _, kv_cache, prefix_mask, prefix_len = self._run_prefix(observation)

        def _step(x_t, time, step_rng):
            time_b = jnp.broadcast_to(time, batch_size)
            suffix_out, _, _ = self._suffix_forward(
                observation, x_t, time_b, kv_cache, prefix_mask, prefix_len
            )  # z=None: matches how the flow-matching action loss was actually trained
            v_bc = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            # NO Python `if` here. bc_velocity_noise crosses the
            # module_jit boundary as a traced value, so `if bc_velocity_noise:` raises
            # TracerBoolConversionError -- which _run_episode_batch then swallows into
            # success=False, producing a clean-looking 0%% success rate. Multiplying by the
            # (possibly-zero) scalar is numerically identical at 0.0 and safe under tracing.
            v_bc = v_bc + jnp.asarray(bc_velocity_noise, v_bc.dtype) * jax.random.normal(step_rng, v_bc.shape)
            # Same no-Python-`if` rule as above: multiply by the (possibly-zero) magnitude rather
            # than branch, so bc_velocity_bias=0.0 is numerically identical and safe under tracing.
            v_bc = v_bc + bias_vec.astype(v_bc.dtype)
            x_hat0 = x_t - time * v_bc
            # F is trained on CLEAN action chunks (t pinned to 0 in the rep loss), so x_hat0 is
            # the in-support query and the noisy iterate x_t is not. `guidance_eval_at_xt=True`
            # queries it anyway, as an ablation: measured ||grad_a Q|| is 168->190 at x_hat0 over
            # t = 0.95->0.05 but only 1.1->26.9 at x_t, so expect it to make guidance inert rather
            # than better. A real noise-conditional value function is a training change.
            # A Python `if` is safe here: eval_at_xt is a concrete bool before tracing.
            grad_eval_point = x_t if eval_at_xt else x_hat0
            grad, q_value = self.grad_a_q(observation, grad_eval_point, kv_cache, prefix_mask, prefix_len, z_r)
            grad_clipped, grad_norm = _clip_guidance_grad(grad, clip_norm)
            # TODO: guidance is currently constant-weight across all t; x_hat0 is a poor one-step
            # estimate early in denoising (large t), so trusting grad_a_q less there (via
            # shape="linear_anneal"/"power_anneal" on GuidanceWeightSchedule, already implemented
            # above) may help -- see the fuller discussion in the TODO on
            # sample_actions_guided_noise_conditional below.
            weight = schedule(time)
            v_guided = v_bc - weight * grad_clipped

            bc_norm = jnp.linalg.norm(v_bc.reshape(batch_size, -1), axis=-1)  # [b], pre-mean
            # The velocity term actually added to v_bc is -weight*grad_clipped, not grad itself, so
            # negate here: alignment near +1 means the ACTUAL guidance contribution reinforces the
            # base policy's direction; near -1 means it opposes it.
            alignment = -_cos_sim(grad, v_bc)
            clip_hit_frac = (
                jnp.mean((grad_norm > clip_norm).astype(jnp.float32)) if clip_norm is not None else jnp.asarray(0.0)
            )
            # Use the clipped norm (what's actually added to v_bc), not the raw pre-clip grad_norm
            # -- otherwise this overstates the true relative magnitude whenever clipping is active.
            grad_norm_clipped = jnp.minimum(grad_norm, clip_norm) if clip_norm is not None else grad_norm
            relative_guidance_magnitude = jnp.mean(weight * grad_norm_clipped / (bc_norm + 1e-8))
            x_hat0_norm = jnp.mean(jnp.linalg.norm(x_hat0.reshape(batch_size, -1), axis=-1))
            q_value_mean = jnp.mean(q_value) if q_value is not None else jnp.asarray(jnp.nan)
            # Sign flipped once more so POSITIVE means "opposes the bias" (see the docstring).
            # Against the unit bias direction and grad_clipped, not the raw grad. Meaningless when
            # bc_velocity_bias == 0.
            bias_flat = bias_dir.reshape(1, -1)
            guid_contrib = (-grad_clipped).reshape(batch_size, -1)
            bias_pushback = -jnp.mean(
                jnp.sum(guid_contrib * bias_flat, axis=-1)
                / (jnp.linalg.norm(guid_contrib, axis=-1) * jnp.linalg.norm(bias_flat) + 1e-8)
            )

            step_stats = (
                weight,
                jnp.mean(grad_norm),
                jnp.mean(bc_norm),
                alignment,
                clip_hit_frac,
                relative_guidance_magnitude,
                x_hat0_norm,
                q_value_mean,
                bias_pushback,
            )
            return x_t + dt * v_guided, time + dt, step_stats

        if not return_diagnostics:

            def cond(carry):
                _, time, _ = carry
                return time >= -dt / 2  # robust to floating-point error

            def step(carry):
                x_t, time, r = carry
                r, sub = jax.random.split(r)
                x_next, time_next, _ = _step(x_t, time, sub)
                return x_next, time_next, r

            x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, bc_noise_rng))
            return x_0

        n_diag = 9  # must match the length of step_stats above

        def body(i, carry):
            x_t, time, r, *hists = carry
            r, sub = jax.random.split(r)
            x_next, time_next, step_stats = _step(x_t, time, sub)
            new_hists = [h.at[i].set(v) for h, v in zip(hists, step_stats, strict=True)]
            return (x_next, time_next, r, *new_hists)

        init = (noise, jnp.asarray(1.0), bc_noise_rng, *[jnp.zeros((num_steps,)) for _ in range(n_diag)])
        result = jax.lax.fori_loop(0, num_steps, body, init)
        x_0 = result[0]
        (
            weight_hist,
            gradnorm_hist,
            bcnorm_hist,
            alignment_hist,
            cliphit_hist,
            relmag_hist,
            xhat0norm_hist,
            qvalue_hist,
            biaspushback_hist,
        ) = result[3:]
        diagnostics = {
            "guidance_weight_per_step": weight_hist,
            "guidance_grad_norm_per_step": gradnorm_hist,
            "bc_velocity_norm_per_step": bcnorm_hist,
            "guidance_alignment_per_step": alignment_hist,
            "guidance_clip_hit_frac_per_step": cliphit_hist,
            "relative_guidance_magnitude_per_step": relmag_hist,
            "x_hat0_norm_per_step": xhat0norm_hist,
            "q_value_per_step": qvalue_hist,
            "q_value_delta": qvalue_hist[-1] - qvalue_hist[0],
            # >0 means guidance opposes the injected bias. Only meaningful
            # when bc_velocity_bias != 0. See the docstring: this, not success rate, is the
            # measurement the fixed-bias control exists to produce.
            "bc_bias_pushback_per_step": biaspushback_hist,
            "bc_bias_pushback_cos": jnp.mean(biaspushback_hist),
            "x0_norm": jnp.mean(jnp.linalg.norm(x_0.reshape(batch_size, -1), axis=-1)),
        }
        return x_0, diagnostics

    def sample_actions_guided_noise_conditional(self, *args, **kwargs):
        """TODO: evaluate grad_a Q at the noisy iterate x_t directly (a noise-conditional
        phi(s, x_t, t, z)) rather than via x_hat_0, which is necessarily poor at large t. Needs
        (a) phi trained with t sampled the way the action loss samples it, instead of pinned to 0,
        and (b) a check that the resulting noisy-Q is calibrated enough to guide with at large t
        -- or that guidance should just anneal off as t -> 1. Empty for now.
        """
        raise NotImplementedError
