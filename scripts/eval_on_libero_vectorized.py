"""Vectorized counterpart to eval_on_libero.py: runs `num_parallel_envs` LIBERO episodes
concurrently (via LIBERO's own `SubprocVectorEnv`, one env per subprocess) to speed up
evaluation, plus a batched policy inference call per step (all currently-active envs'
observations go through the model in ONE forward pass instead of one call per env).

Identical CLI surface, metrics schema and wandb logging to eval_on_libero.py -- `CheckpointSource`,
`EpisodeResult`, `SUITE_MAX_STEPS`, `_get_libero_env`, `_obs_to_element`, `_save_episode_video`,
`_aggregate_and_log` and `Args` are reused from that module rather than duplicated. Only HOW
rollouts are collected differs. Same JAX+LIBERO dependency requirement; run via
`uv run --group libero scripts/eval_on_libero_vectorized.py ...`.

Two correctness notes:

  1. EGL headless rendering: `MUJOCO_GL`/`PYOPENGL_PLATFORM` must be "egl" before ANY import that
     could transitively pull in mujoco/robosuite, hence the very first lines of this file.
  2. Multiprocessing start method: `SubprocVectorEnv` uses the process-global default, `fork` on
     Linux, and forking a process holding a CUDA context is a known deadlock/corruption hazard.
     We force `spawn` at import time, and as defense in depth build one throwaway SubprocVectorEnv
     in `main()` BEFORE the JAX policy is loaded.

Usage:
uv run --group libero scripts/eval_on_libero_vectorized.py \
    --args.checkpoint.config=pi05_libero \
    --args.checkpoint.dir=checkpoints/pi05_libero/my_experiment/20000 \
    --args.task-suites libero_spatial libero_object \
    --args.num-trials-per-task=10 \
    --args.num-parallel-envs=4
"""

# ruff: noqa: E402 -- the env-var/start-method setup below MUST run before any import that could
# transitively pull in mujoco/robosuite/JAX (see module docstring points 1-2), so the usual
# imports-at-top-of-file rule doesn't apply here.
import logging
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _resolve_mujoco_egl_device_id() -> None:
    """Sets MUJOCO_EGL_DEVICE_ID to the EGL device index that actually corresponds to the GPU(s)
    exposed by CUDA_VISIBLE_DEVICES.

    Without MUJOCO_EGL_DEVICE_ID, robosuite's picker indexes EGL's device list with the raw
    CUDA_VISIBLE_DEVICES value. EGL enumeration is a separate driver API whose ordering need not
    match CUDA's -- on this machine EGL index i is CUDA ordinal (3 - i), so CUDA_VISIBLE_DEVICES=1
    silently rendered on GPU 2.

    This queries the true mapping via the EGL_NV_device_cuda extension in this already-restricted
    process. `eglQueryDevicesEXT` still enumerates every raw device, but the CUDA-ordinal query
    succeeds only for the exposed ones and renumbers them from 0, so we look for whichever device
    resolves to ordinal 0. Leaves MUJOCO_EGL_DEVICE_ID unset (robosuite's original behaviour) if
    CUDA_VISIBLE_DEVICES is unset, the variable is already set, or the extension is unavailable.
    """
    if "MUJOCO_EGL_DEVICE_ID" in os.environ or not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return

    try:
        import ctypes

        from mujoco.egl import egl_ext as _egl_ext

        devices = _egl_ext.eglQueryDevicesEXT()
        query_attrib = ctypes.CFUNCTYPE(
            _egl_ext.EGLBoolean, _egl_ext.EGLDeviceEXT, _egl_ext.EGLint, ctypes.POINTER(ctypes.c_ssize_t)
        )(_egl_ext.eglGetProcAddress("eglQueryDeviceAttribEXT"))
        egl_cuda_device_nv = 0x323A  # EGL_CUDA_DEVICE_NV, from the EGL_NV_device_cuda extension

        matches = []
        for egl_index, device in enumerate(devices):
            ordinal = ctypes.c_ssize_t(-1)
            if query_attrib(device, egl_cuda_device_nv, ctypes.byref(ordinal)) and ordinal.value == 0:
                matches.append(egl_index)

        if len(matches) != 1:
            logging.warning(
                f"Expected exactly one EGL device to resolve to CUDA ordinal 0 under "
                f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}, found {matches}; "
                "leaving MUJOCO_EGL_DEVICE_ID unset."
            )
            return
        egl_index = matches[0]
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_index)
        logging.info(
            f"Resolved CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r} -> "
            f"MUJOCO_EGL_DEVICE_ID={egl_index} (EGL_NV_device_cuda)"
        )
    except Exception:
        logging.exception(
            "Failed to resolve MUJOCO_EGL_DEVICE_ID via EGL_NV_device_cuda; leaving it unset "
            "(falls back to robosuite's CUDA_VISIBLE_DEVICES-as-raw-index behavior, which may "
            "render on the wrong physical GPU on multi-GPU machines)."
        )


_resolve_mujoco_egl_device_id()

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import collections
import dataclasses
import pathlib
from typing import Protocol

import eval_on_libero as eval_lib
from libero.libero import benchmark
from libero.libero.envs import SubprocVectorEnv
import numpy as np
import tqdm
import tyro
import wandb

LIBERO_DUMMY_ACTION = eval_lib.LIBERO_DUMMY_ACTION
LIBERO_ENV_RESOLUTION = eval_lib.LIBERO_ENV_RESOLUTION
SUITE_MAX_STEPS = eval_lib.SUITE_MAX_STEPS


class _InferBatchPolicy(Protocol):
    """The interface this script relies on beyond `_InferPolicy` -- an in-process `Policy` (see
    `openpi.policies.policy.Policy.infer_batch`) satisfies this. Purely additive on the `Policy`
    side: batched inference inherently requires an in-process model, so `WebsocketClientPolicy`
    (which only implements `infer`) cannot be used with this script."""

    def infer_batch(self, obs_list: list[dict]) -> list[dict]: ...


def _make_env_factory(task, resolution: int, seed: int):
    """Returns a zero-arg, picklable callable that builds and seeds one LIBERO env, for use as a
    SubprocVectorEnv env_fn (wraps eval_lib._get_libero_env, discarding its description return
    value since callers already have `task.language`)."""

    def _factory():
        env, _ = eval_lib._get_libero_env(task, resolution, seed)  # noqa: SLF001 -- intentional cross-module reuse
        return env

    return _factory


def _run_episode_batch(
    policy: _InferBatchPolicy,
    venv: SubprocVectorEnv,
    env_ids: list[int],
    task_description: str,
    initial_states: list[np.ndarray],
    *,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    step_pbar: tqdm.tqdm | None = None,
    infer_batch_fn=None,
) -> tuple[list[tuple[bool, int, list[np.ndarray], str | None]], list[dict]]:
    """Batched counterpart to eval_lib._run_episode. Runs len(env_ids) episodes in lockstep on
    `venv`, one per env slot in `env_ids`, preserving PER-ENV action-chunk replanning (one
    collections.deque per slot), per-env wait-phase, and per-env early termination on `done`.
    Returns (per_env_results, batch_diagnostics):
      - per_env_results: list of (success, steps_taken, replay_images, error), one per env_id, in
        the same order as `env_ids` (== the order of `initial_states`). `error` is None for an
        episode that reached a genuine outcome and a message for one aborted by an exception --
        see the except block; these must not be pooled into a success rate.
      - batch_diagnostics: FLAT list of `out["guidance_diagnostics"]` dicts, collected ONCE per
        `infer_batch()` call (not once per env) -- a single batched infer_batch call already
        produces one shared, batch-averaged diagnostics dict for every example in it (see
        guided_policy.py's `infer_batch`), so collecting it per-env here would double/triple-count
        the same numbers when this gets aggregated into per-task wandb metrics.

    `infer_batch_fn`, if given, replaces `policy.infer_batch` as the action-chunk source (e.g.
    `policy.infer_batch_unguided` for TTA probe rollouts) -- defaults to `policy.infer_batch`.

    Exception handling is at BATCH granularity (a failure ends the whole batch, not just one
    env) -- true per-episode isolation would require falling back to per-env venv.step(id=[...])
    calls, defeating the point of batching. Accepted v1 limitation.
    """
    infer_batch = infer_batch_fn if infer_batch_fn is not None else policy.infer_batch
    venv.reset(id=env_ids)
    action_plans: dict[int, collections.deque] = {eid: collections.deque() for eid in env_ids}
    obs_stack = venv.set_init_state(init_state=initial_states, id=env_ids)
    obs_by_id = {eid: obs_stack[k] for k, eid in enumerate(env_ids)}

    replay_images_by_id: dict[int, list[np.ndarray]] = {eid: [] for eid in env_ids}
    # None means "this episode ran to a real conclusion". See the except
    # block at the bottom of the loop.
    error_by_id: dict[int, str | None] = dict.fromkeys(env_ids)
    steps_taken_by_id: dict[int, int] = dict.fromkeys(env_ids, 0)
    success_by_id: dict[int, bool] = dict.fromkeys(env_ids, False)
    batch_diagnostics: list[dict] = []

    active_ids = list(env_ids)
    t = 0
    try:
        while t < max_steps + num_steps_wait and active_ids:
            if t < num_steps_wait:
                obs_stack, _, _, _ = venv.step([LIBERO_DUMMY_ACTION] * len(active_ids), id=active_ids)
                for k, eid in enumerate(active_ids):
                    obs_by_id[eid] = obs_stack[k]
                t += 1
                if step_pbar is not None:
                    step_pbar.update(1)
                continue

            elements = {}
            for eid in active_ids:
                element, display_img = eval_lib._obs_to_element(  # noqa: SLF001 -- intentional cross-module reuse
                    obs_by_id[eid], task_description, resize_size
                )
                elements[eid] = element
                replay_images_by_id[eid].append(display_img)

            need_infer_ids = [eid for eid in active_ids if not action_plans[eid]]
            if need_infer_ids:
                outputs = infer_batch([elements[eid] for eid in need_infer_ids])
                if outputs and "guidance_diagnostics" in outputs[0]:
                    batch_diagnostics.append(outputs[0]["guidance_diagnostics"])
                for eid, out in zip(need_infer_ids, outputs, strict=True):
                    action_chunk = out["actions"]
                    assert len(action_chunk) >= replan_steps, (
                        f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    )
                    action_plans[eid].extend(action_chunk[:replan_steps])

            actions_this_step = [action_plans[eid].popleft().tolist() for eid in active_ids]
            obs_stack, _, done, _ = venv.step(actions_this_step, id=active_ids)
            for k, eid in enumerate(active_ids):
                obs_by_id[eid] = obs_stack[k]
                steps_taken_by_id[eid] += 1
            t += 1
            if step_pbar is not None:
                step_pbar.update(1)

            newly_done = [eid for eid, d in zip(active_ids, done, strict=True) if d]
            for eid in newly_done:
                success_by_id[eid] = True
            active_ids = [eid for eid in active_ids if eid not in newly_done]
    except Exception as exc:
        logging.exception(
            f"Episode batch failed for task {task_description!r} (env_ids={env_ids}, still-active={active_ids})"
        )
        # record WHICH episodes were cut short. Everything still active when
        # the batch died has an unknown outcome, not a failed one -- but the code below reports
        # success_by_id[eid] == False for them either way. Left unlabeled, a transient EGL or
        # mujoco fault (much likelier under parallel envs) is indistinguishable from the policy
        # genuinely failing the task, and a crashed cell reads as a confident 0% success rate.
        # Episodes that already finished before the exception keep their real result.
        batch_error = f"{type(exc).__name__}: {exc}"
        for eid in active_ids:
            error_by_id[eid] = batch_error

    per_env_results = [
        (success_by_id[eid], steps_taken_by_id[eid], replay_images_by_id[eid], error_by_id[eid]) for eid in env_ids
    ]
    return per_env_results, batch_diagnostics


def _run_tta_probe_episode_batch(
    policy: _InferBatchPolicy,
    venv: SubprocVectorEnv,
    env_ids: list[int],
    task_description: str,
    initial_states: list[np.ndarray],
    *,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    tta_sample_stride: int,
    dense_reward_gamma: float | None = None,
    step_pbar: tqdm.tqdm | None = None,
) -> list[tuple[dict, float]]:
    """Batched counterpart to eval_lib._run_tta_probe_episode. Runs len(env_ids) UNGUIDED probe
    episodes in lockstep on `venv` (via `policy.infer_batch_unguided`), collecting (element,
    reward) pairs from each env's own native `env.step` reward, per-env, every
    `tta_sample_stride`-th post-warmup step AND always on that env's terminal/`done` step. Returns
    a flat, pooled list across all env_ids (order doesn't matter -- these just feed test_time_adapt
    as an unordered set of samples).
    """
    venv.reset(id=env_ids)
    action_plans: dict[int, collections.deque] = {eid: collections.deque() for eid in env_ids}
    obs_stack = venv.set_init_state(init_state=initial_states, id=env_ids)
    obs_by_id = {eid: obs_stack[k] for k, eid in enumerate(env_ids)}
    step_count_by_id: dict[int, int] = dict.fromkeys(env_ids, 0)

    collected: list[tuple[dict, float]] = []
    active_ids = list(env_ids)
    t = 0
    try:
        while t < max_steps + num_steps_wait and active_ids:
            if t < num_steps_wait:
                obs_stack, _, _, _ = venv.step([LIBERO_DUMMY_ACTION] * len(active_ids), id=active_ids)
                for k, eid in enumerate(active_ids):
                    obs_by_id[eid] = obs_stack[k]
                t += 1
                if step_pbar is not None:
                    step_pbar.update(1)
                continue

            elements = {}
            for eid in active_ids:
                element, _ = eval_lib._obs_to_element(  # noqa: SLF001 -- intentional cross-module reuse
                    obs_by_id[eid], task_description, resize_size
                )
                elements[eid] = element

            need_infer_ids = [eid for eid in active_ids if not action_plans[eid]]
            if need_infer_ids:
                outputs = policy.infer_batch_unguided([elements[eid] for eid in need_infer_ids])
                for eid, out in zip(need_infer_ids, outputs, strict=True):
                    action_plans[eid].extend(out["actions"][:replan_steps])

            actions_this_step = [action_plans[eid].popleft().tolist() for eid in active_ids]
            obs_stack, rewards, done, _ = venv.step(actions_this_step, id=active_ids)
            for k, eid in enumerate(active_ids):
                obs_by_id[eid] = obs_stack[k]
                step_count_by_id[eid] += 1
                if step_count_by_id[eid] % tta_sample_stride == 0 or done[k]:
                    # eid/step recorded so dense shaping can reconstruct time-to-success; the
                    # samples are interleaved across parallel envs, so position in `collected`
                    # carries no per-episode timing on its own.
                    collected.append((elements[eid], float(rewards[k]), eid, step_count_by_id[eid]))
            t += 1
            if step_pbar is not None:
                step_pbar.update(1)

            newly_done = [eid for eid, d in zip(active_ids, done, strict=True) if d]
            active_ids = [eid for eid in active_ids if eid not in newly_done]
    except Exception:
        logging.exception(
            f"TTA probe batch failed for task {task_description!r} (env_ids={env_ids}, still-active={active_ids})"
        )
    return _shape_probe_rewards(collected, dense_reward_gamma)


def _shape_probe_rewards(
    collected: list[tuple[dict, float, int, int]], dense_reward_gamma: float | None
) -> list[tuple[dict, float]]:
    """Optional dense reward shaping by time-to-success.

    LIBERO's reward fires on exactly ONE frame, so a successful probe contributes exactly one
    vector to z_r = E[r * psi] -- median effective sample size across 32 tasks is 2.5.

    With `dense_reward_gamma = g`, every frame of a SUCCESSFUL episode at env-step t gets
    r = g^(T - t) for that episode's success step T; frames after T and all frames of failed
    episodes stay 0. One informative sample per success becomes ~40.

    This is a surrogate -- "discounted proximity to success", not the success indicator -- and a
    gain here does not by itself vindicate the sparse formulation. g is per ENV STEP: at g=0.98 a
    sample 100 steps before success carries 0.13 and 400 steps before carries 0.0003.
    Pass None to keep the original sparse behavior exactly.
    """
    if dense_reward_gamma is None:
        return [(el, r) for el, r, _, _ in collected]
    success_step: dict[int, int] = {}
    for _, r, eid, step in collected:
        if r > 0.0 and eid not in success_step:
            success_step[eid] = step
    out: list[tuple[dict, float]] = []
    for el, _r, eid, step in collected:
        t_succ = success_step.get(eid)
        if t_succ is None or step > t_succ:
            out.append((el, 0.0))
        else:
            out.append((el, float(dense_reward_gamma ** (t_succ - step))))
    return out


def _select_guidance_scale_by_probe(
    policy,
    venv,
    env_ids_cap: int,
    task_description: str,
    probe_states,
    candidates,
    *,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    step_pbar=None,
) -> tuple[float, dict[float, float]]:
    """Choose this task's guidance scale from GUIDED probe rollouts.

    The best scale varies enormously by task (measured optima of 1.0, 2.0, 5.0 and >20 across ten
    tasks), so a single global scale leaves most of the gain unclaimed -- but picking it from the
    evaluation episodes would be selection on the test set, worth ~+0.076 of pure artifact by
    permutation null. Probes run on the initial states reserved for TTA, which the evaluation never
    scores, so the tuning and reported episodes are disjoint.

    Requires z_r to be set already (the probes are guided). Returns (best_scale, {scale: success}).
    """
    from openpi.models.rep_grad_guidance import GuidanceWeightSchedule

    scores: dict[float, float] = {}
    for scale in candidates:
        policy.set_sample_kwarg("guidance_schedule", GuidanceWeightSchedule(base_weight=float(scale)))
        wins = n = 0
        idx = 0
        while idx < len(probe_states):
            bs = min(env_ids_cap, len(probe_states) - idx)
            ids = list(range(bs))
            batch = [probe_states[idx + k] for k in range(bs)]
            if step_pbar is not None:
                step_pbar.reset(total=max_steps + num_steps_wait)
                step_pbar.set_description(f"scale-probe g={scale} {idx}-{idx + bs - 1}")
            res, _ = _run_episode_batch(
                policy,
                venv,
                ids,
                task_description,
                batch,
                resize_size=resize_size,
                replan_steps=replan_steps,
                num_steps_wait=num_steps_wait,
                max_steps=max_steps,
                step_pbar=step_pbar,
            )
            for success, _steps, _imgs, err in res:
                # An errored probe is not evidence about the scale; drop it rather than score it
                # as a failure, which would bias selection toward whichever scale crashed least.
                if err is None:
                    wins += int(success)
                    n += 1
            idx += bs
        scores[float(scale)] = wins / n if n else float("nan")
    # Ties -> smallest scale: the least intervention that achieves the best observed probe score,
    # which is also the safer extrapolation to unseen initial states.
    best = max(sorted(scores), key=lambda s_: (scores[s_] if scores[s_] == scores[s_] else -1.0))
    policy.set_sample_kwarg("guidance_schedule", GuidanceWeightSchedule(base_weight=float(best)))
    return best, scores


def run_libero_eval_vectorized(
    policy: _InferBatchPolicy,
    *,
    task_suites: tuple[str, ...] = ("libero_spatial",),
    task_ids: dict[str, tuple[int, ...] | None] | tuple[int, ...] | None = None,
    num_trials_per_task: int = 50,
    num_parallel_envs: int = 10,
    resize_size: int = 224,
    replan_steps: int = 5,
    num_steps_wait: int = 10,
    seed: int = 7,
    video_out_dir: str | pathlib.Path = "data/libero/videos",
    log_videos_to_wandb: bool = True,
    max_videos_per_task: int = 5,
    wandb_enabled: bool = True,
    wandb_run_id: str | None = None,
    wandb_project: str | None = None,
    enable_guided_eval: bool = True,
    num_tta_episodes: int = 10,
    tta_sample_stride: int = 10,
    tta_repeats: int = 1,
    max_tta_samples: int = 1_000_000,
    metric_prefix: str = "eval",
    zr_whiten_ridge: float | None = None,
    center_reward: bool = False,
    dense_reward_gamma: float | None = None,
    bc_velocity_noise: float = 0.0,
    bc_velocity_bias: float = 0.0,
    num_denoise_steps: int | None = None,
    per_episode_rng: bool = True,
    results_json: str | pathlib.Path | None = None,
    run_args: dict | None = None,
) -> dict:
    """Vectorized counterpart to eval_lib.run_libero_eval -- same semantics, metrics schema, and
    wandb logging (via the shared `eval_lib._aggregate_and_log`/`_save_episode_video`), but runs
    up to `num_parallel_envs` episodes concurrently per task via SubprocVectorEnv, with batched
    policy inference across all currently-active envs each step.

    `metric_prefix` (default "eval") replaces the leading "eval" in every wandb key this run logs
    -- see `eval_lib.run_libero_eval`'s docstring for why (running this script multiple times
    against different checkpoints of the SAME resumed wandb run).

    Episodes are NOT guaranteed bit-identical to the sequential script (each parallel env slot
    gets its own seed, `seed + slot_index`, matching SubprocVectorEnv's own seed() broadcast
    convention) -- only structurally comparable (same metrics schema, plausible distributions).

    Two-phase evaluation for value-guided-denoising policies -- see eval_lib.run_libero_eval's own
    docstring for the full rationale; here, TTA probe batches reuse the same `venv` pool as the
    real episodes for that task, via `policy.infer_batch_unguided`/`_run_tta_probe_episode_batch`.
    """
    if wandb_enabled:
        if wandb_run_id is not None:
            wandb.init(id=wandb_run_id, resume="must", project=wandb_project)
        else:
            logging.warning("No wandb_run_id given; starting a fresh standalone wandb run.")
            wandb.init(project=wandb_project or "libero_eval")

        guidance_model = getattr(policy, "_model", None)
        guidance_schedule = getattr(guidance_model, "guidance_schedule", None)
        if guidance_schedule is not None:
            wandb.config.update(
                {
                    f"{metric_prefix}_guidance_base_weight": guidance_schedule.base_weight,
                    f"{metric_prefix}_guidance_schedule_shape": guidance_schedule.shape,
                    f"{metric_prefix}_guidance_grad_clip_norm": getattr(
                        guidance_model, "guidance_grad_clip_norm", None
                    ),
                    f"{metric_prefix}_normalize_z_r": getattr(guidance_model, "normalize_z_r_default", None),
                },
                allow_val_change=True,
            )

    video_out_dir = pathlib.Path(video_out_dir)
    video_out_dir.mkdir(parents=True, exist_ok=True)
    # degradation for the positive control. Applied to the policy so both
    # the guided and scale-0 arms see it -- the comparison isolates guidance, not the noise.
    if bc_velocity_noise and hasattr(policy, "set_sample_kwarg"):
        policy.set_sample_kwarg("bc_velocity_noise", bc_velocity_noise)
        logging.info(f"Degrading BC velocity with sigma={bc_velocity_noise} (positive control)")
    # The fixed-bias arm of the positive control: unlike the noise above it does not average out
    # across denoising steps, so it is the degradation that actually moves success.
    # Second degradation axis. A constant velocity bias can be absorbed by the closed loop --
    # on libero_spatial bias 0.5 leaves success at 1.000 where it costs libero_90 0.47 -> 0.42.
    # Under-solving the flow ODE cannot be averaged away: every replan re-commits the SAME
    # systematic integration error, off the BC manifold in a direction guidance operates in.
    if num_denoise_steps is not None and hasattr(policy, "set_sample_kwarg"):
        policy.set_sample_kwarg("num_steps", num_denoise_steps)
        logging.info(f"Under-solving the flow ODE with num_steps={num_denoise_steps} (positive control)")
    if bc_velocity_bias and hasattr(policy, "set_sample_kwarg"):
        policy.set_sample_kwarg("bc_velocity_bias", bc_velocity_bias)
        logging.info(f"Degrading BC velocity with FIXED bias={bc_velocity_bias} (positive control)")

    benchmark_dict = benchmark.get_benchmark_dict()
    results: list[eval_lib.EpisodeResult] = []
    # per-task diagnostics accumulated for the results JSON, keyed by
    # (suite, task_id). Previously these went straight to wandb and were unrecoverable offline.
    tta_by_task: dict[tuple[str, int], dict] = {}
    guidance_by_task: dict[tuple[str, int], dict] = {}

    suite_pbar = tqdm.tqdm(total=len(task_suites), desc="suites", position=0, leave=True)
    task_pbar = tqdm.tqdm(total=0, desc="tasks", position=1, leave=True)
    episode_pbar = tqdm.tqdm(total=num_trials_per_task, desc="episode-batches", position=2, leave=True)
    step_pbar = tqdm.tqdm(total=0, desc="steps", position=3, leave=True)

    try:
        for suite_name in task_suites:
            if suite_name not in SUITE_MAX_STEPS:
                raise ValueError(f"Unknown task suite: {suite_name!r}. Options: {sorted(SUITE_MAX_STEPS)}")
            task_suite = benchmark_dict[suite_name]()
            max_steps = SUITE_MAX_STEPS[suite_name]

            suite_task_ids = task_ids.get(suite_name) if isinstance(task_ids, dict) else task_ids
            ids_to_run = list(range(task_suite.n_tasks)) if suite_task_ids is None else list(suite_task_ids)

            suite_pbar.set_description(f"suite={suite_name}")
            task_pbar.reset(total=len(ids_to_run))

            for task_id in ids_to_run:
                task = task_suite.get_task(task_id)
                task_description = task.language
                initial_states = task_suite.get_task_init_states(task_id)

                b_task = min(num_parallel_envs, num_trials_per_task)
                env_fns = [_make_env_factory(task, LIBERO_ENV_RESOLUTION, seed + slot) for slot in range(b_task)]
                venv = SubprocVectorEnv(env_fns)

                task_pbar.set_description(f"[{suite_name}] task {task_id}: {task_description[:40]}")
                task_z_r_norm = float("nan")

                try:
                    active_infer_batch = policy.infer_batch
                    if enable_guided_eval and hasattr(policy, "test_time_adapt") and hasattr(policy, "set_z_r"):
                        if num_tta_episodes > 0:
                            n_avail = len(initial_states)
                            if num_tta_episodes + num_trials_per_task > n_avail:
                                logging.warning(
                                    f"[{suite_name}#{task_id}] num_tta_episodes({num_tta_episodes}) + "
                                    f"num_trials_per_task({num_trials_per_task}) > available init "
                                    f"states ({n_avail}); TTA probes may reuse real-eval initial states."
                                )
                            tta_initial_states = (
                                initial_states[-num_tta_episodes:] if num_tta_episodes <= n_avail else initial_states
                            )
                            # Raise the probe budget WITHOUT stealing initial states from the
                            # real eval: libero_90 exposes 50 init states per task and 40 trials +
                            # 10 probes consumes all of them, so a larger num_tta_episodes would
                            # fit z_r on the test set. Replaying the same reserved states attacks
                            # estimator variance from policy stochasticity (~2.5 informative
                            # samples at 10 probes, ~15 at 60) but adds no initial-state
                            # diversity, so tasks the policy never solves stay at ESS 0.
                            tta_initial_states = list(tta_initial_states) * max(1, tta_repeats)
                            # probes get their own deterministic seed
                            # so z_r does not depend on execution history either.
                            if per_episode_rng and hasattr(policy, "reset_rng"):
                                policy.reset_rng(seed * 7_919 + task_id)
                            probe_samples: list[tuple[dict, float]] = []
                            tta_idx = 0
                            while tta_idx < len(tta_initial_states):
                                tta_batch_size = min(b_task, len(tta_initial_states) - tta_idx)
                                tta_env_ids = list(range(tta_batch_size))
                                tta_batch_states = [tta_initial_states[tta_idx + k] for k in range(tta_batch_size)]
                                step_pbar.reset(total=max_steps + num_steps_wait)
                                step_pbar.set_description(
                                    f"[{suite_name}#{task_id}] TTA probes {tta_idx}-{tta_idx + tta_batch_size - 1}"
                                )
                                task_pbar.set_description(
                                    f"[{suite_name}#{task_id}] TTA probes {tta_idx}-{tta_idx + tta_batch_size - 1}"
                                )
                                probe_samples.extend(
                                    _run_tta_probe_episode_batch(
                                        policy,
                                        venv,
                                        tta_env_ids,
                                        task_description,
                                        tta_batch_states,
                                        resize_size=resize_size,
                                        replan_steps=replan_steps,
                                        num_steps_wait=num_steps_wait,
                                        max_steps=max_steps,
                                        tta_sample_stride=tta_sample_stride,
                                        dense_reward_gamma=dense_reward_gamma,
                                        step_pbar=step_pbar,
                                    )
                                )
                                tta_idx += tta_batch_size

                            # stratified, so the rare nonzero-reward
                            # samples (the only ones z_r actually depends on) survive the cap.
                            probe_samples = eval_lib.subsample_probe_samples(probe_samples, max_tta_samples)

                            if probe_samples:
                                elements = [e for e, _ in probe_samples]
                                rewards = np.array([r for _, r in probe_samples], dtype=np.float32)
                                reward_samples = policy.build_reward_samples(elements, rewards)
                                z_r = policy.test_time_adapt(
                                    reward_samples, zr_whiten_ridge=zr_whiten_ridge, center_reward=center_reward
                                )
                                z_r_norm = float(np.linalg.norm(z_r))
                                task_z_r_norm = z_r_norm
                                tta_diag = getattr(policy, "_last_tta_diagnostics", {})
                                # keep ALL of _last_tta_diagnostics, not
                                # just the four keys the wandb block below cherry-picks.
                                # frac_nonzero_reward and nonzero_psi_pairwise_consistency are
                                # what distinguish "too few successful probe episodes" from
                                # "the successful probes disagree about direction" -- different
                                # failure modes of z_r = E[r*psi(s)] with different fixes.
                                tta_by_task[(suite_name, task_id)] = {
                                    **tta_diag,
                                    "z_r_norm": z_r_norm,
                                    "num_probe_samples": len(probe_samples),
                                    "reward_mean": float(rewards.mean()),
                                    "reward_max": float(rewards.max()),
                                }
                                logging.info(
                                    f"[{suite_name}#{task_id}] TTA: {len(probe_samples)} samples, "
                                    f"reward mean={rewards.mean():.3f} max={rewards.max():.3f}, "
                                    f"z_r_norm={z_r_norm:.4f}"
                                )
                                if wandb_enabled:
                                    tta_prefix = f"{metric_prefix}/tta/{suite_name}/task{task_id}"
                                    wandb.log(
                                        {
                                            f"{tta_prefix}/num_samples": len(probe_samples),
                                            f"{tta_prefix}/z_r_norm": z_r_norm,
                                            f"{tta_prefix}/reward_mean": float(rewards.mean()),
                                            f"{tta_prefix}/reward_max": float(rewards.max()),
                                            f"{tta_prefix}/any_nonzero_reward": bool(rewards.max() > 0),
                                            f"{tta_prefix}/z_r_norm_raw": tta_diag.get("z_r_norm_raw", float("nan")),
                                            f"{tta_prefix}/num_nonzero_reward_samples": tta_diag.get(
                                                "num_nonzero_reward_samples", float("nan")
                                            ),
                                            f"{tta_prefix}/effective_sample_size": tta_diag.get(
                                                "effective_sample_size", float("nan")
                                            ),
                                            f"{tta_prefix}/nonzero_psi_pairwise_consistency": tta_diag.get(
                                                "nonzero_psi_pairwise_consistency", float("nan")
                                            ),
                                        }
                                    )
                                active_infer_batch = policy.infer_batch  # now guided, since z_r is set
                            else:
                                logging.warning(
                                    f"[{suite_name}#{task_id}] TTA collected 0 samples; using unguided eval."
                                )
                                active_infer_batch = policy.infer_batch_unguided
                        else:
                            logging.info(
                                f"[{suite_name}#{task_id}] num_tta_episodes=0; guided eval requested "
                                "but no z_r estimated -- using unguided eval for this task."
                            )
                            active_infer_batch = policy.infer_batch_unguided

                    episode_pbar.reset(total=num_trials_per_task)
                    task_guidance_diagnostics: list[dict] = []
                    episode_idx = 0
                    while episode_idx < num_trials_per_task:
                        batch_size = min(b_task, num_trials_per_task - episode_idx)
                        env_ids = list(range(batch_size))
                        batch_initial_states = [initial_states[episode_idx + k] for k in range(batch_size)]

                        step_pbar.reset(total=max_steps + num_steps_wait)
                        step_pbar.set_description(
                            f"[{suite_name}#{task_id}] eps {episode_idx}-{episode_idx + batch_size - 1}"
                        )
                        episode_pbar.set_description(
                            f"[{suite_name}#{task_id}] episodes {episode_idx}-{episode_idx + batch_size - 1}"
                        )

                        # re-seed per batch so the action noise for a
                        # given (task, episode) is independent of everything that ran before it.
                        # Without this the RNG position depends on the cumulative infer count, so
                        # one episode ending a step early re-randomizes every later episode --
                        # which made four identical runs disagree on 14.5% of episodes, more than
                        # the treatment moves. Derived from the run seed so it stays reproducible.
                        if per_episode_rng and hasattr(policy, "reset_rng"):
                            policy.reset_rng(seed * 1_000_003 + task_id * 1_009 + episode_idx)
                        batch_results, batch_diagnostics = _run_episode_batch(
                            policy,
                            venv,
                            env_ids,
                            task_description,
                            batch_initial_states,
                            resize_size=resize_size,
                            replan_steps=replan_steps,
                            num_steps_wait=num_steps_wait,
                            max_steps=max_steps,
                            step_pbar=step_pbar,
                            infer_batch_fn=active_infer_batch,
                        )
                        task_guidance_diagnostics.extend(batch_diagnostics)
                        # Shared across every env in this batch -- batch_diagnostics is already a
                        # batch-averaged summary, not per-env, see _run_episode_batch's docstring.
                        batch_guidance_summary = eval_lib._summarize_episode_diagnostics(  # noqa: SLF001
                            batch_diagnostics
                        )
                        if batch_guidance_summary is not None:
                            batch_guidance_summary["z_r_norm"] = task_z_r_norm

                        for k, (success, steps_taken, replay_images, ep_error) in enumerate(batch_results):
                            ep_idx = episode_idx + k
                            video_path = eval_lib._save_episode_video(  # noqa: SLF001 -- intentional cross-module reuse
                                replay_images,
                                suite_name=suite_name,
                                task_id=task_id,
                                task_description=task_description,
                                episode_idx=ep_idx,
                                success=success,
                                video_out_dir=video_out_dir,
                                wandb_enabled=wandb_enabled,
                                log_videos_to_wandb=log_videos_to_wandb,
                                max_videos_per_task=max_videos_per_task,
                                metric_prefix=metric_prefix,
                            )
                            results.append(
                                eval_lib.EpisodeResult(
                                    suite_name,
                                    task_id,
                                    task_description,
                                    ep_idx,
                                    success,
                                    steps_taken,
                                    video_path,
                                    batch_guidance_summary,
                                    # initial_states is indexed by ep_idx
                                    # directly (see batch_initial_states above), and the env seed
                                    # follows SubprocVectorEnv's slot convention. Recorded so a
                                    # paired comparison across cells can be verified, not assumed.
                                    initial_state_index=ep_idx,
                                    env_seed=seed + (ep_idx % b_task),
                                    error=ep_error,
                                )
                            )

                        episode_pbar.update(batch_size)
                        episode_idx += batch_size

                    task_guidance = eval_lib._log_task_guidance_diagnostics(  # noqa: SLF001 -- intentional cross-module reuse
                        suite_name,
                        task_id,
                        task_guidance_diagnostics,
                        wandb_enabled=wandb_enabled,
                        metric_prefix=metric_prefix,
                    )
                    if task_guidance is not None:
                        guidance_by_task[(suite_name, task_id)] = task_guidance
                finally:
                    venv.close()

                task_pbar.update(1)
            suite_pbar.update(1)
    finally:
        step_pbar.close()
        episode_pbar.close()
        task_pbar.close()
        suite_pbar.close()

    metrics = eval_lib._aggregate_and_log(  # noqa: SLF001 -- intentional cross-module reuse
        results, wandb_enabled=wandb_enabled, metric_prefix=metric_prefix
    )
    # write the JSON BEFORE wandb.finish(). wandb's final sync can hang or
    # fail on a shared filesystem, and a lost result file is far more expensive than a lost
    # dashboard -- the JSON is what the statistics run on.
    if results_json is not None:
        eval_lib.write_results_json(
            results_json,
            results=results,
            metrics=metrics,
            run_args=run_args or {},
            tta_by_task=tta_by_task,
            guidance_by_task=guidance_by_task,
        )
    if wandb_enabled:
        wandb.finish()
    return metrics


@dataclasses.dataclass
class Args:
    # Policy source. Required for CLI use; for programmatic use, call
    # run_libero_eval_vectorized(...) directly with an already-built policy object instead of
    # going through main().
    checkpoint: eval_lib.CheckpointSource

    #################################################################################################################
    # LIBERO environment-specific parameters (mirrors main.py's Args)
    #################################################################################################################
    task_suites: tuple[str, ...] = (
        "libero_spatial",
    )  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    task_ids: tuple[int, ...] | None = None  # Task indices to run per suite. None = every task in each suite.
    num_trials_per_task: int = 25  # Number of rollouts per task
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    seed: int = 7

    #################################################################################################################
    # Vectorization
    #################################################################################################################
    # Number of LIBERO envs to run concurrently per task (one subprocess each).
    num_parallel_envs: int = 3  # can't go any more than this on the MIDI machines

    #################################################################################################################
    # Output / logging
    #################################################################################################################
    video_out_dir: str = "data/libero/videos"
    log_videos_to_wandb: bool = True
    max_videos_per_task: int = 5  # cap wandb video uploads per task; local .mp4s are always saved

    wandb_enabled: bool = True
    # Wandb run id to attach eval logs to. If not set (and resume_wandb_run is True), auto-derived
    # from <checkpoint_dir's run directory>/wandb_id.txt (the file scripts/train.py itself writes).
    wandb_run_id: str | None = None
    # If True (default), auto-resume the training run found via wandb_id.txt (or wandb_run_id, if
    # explicitly given) so eval metrics land in the SAME run as the checkpoint. Set False to always
    # start a fresh, standalone eval run instead -- e.g. for a one-off eval sweep you don't want
    # mixed into the training run's own logs. Has no effect if wandb_run_id is explicitly set.
    resume_wandb_run: bool = True
    # Wandb project. If not set, derived from the training config's own project_name.
    wandb_project: str | None = None
    # Prefix for every wandb metric key this run logs (default "eval" -> "eval/overall/...",
    # "eval/tta/...", "eval/guidance/...", "eval/videos/...", etc). Override when running this
    # script multiple times against different checkpoints of the SAME (resumed) wandb run -- e.g.
    # --args.metric-prefix=eval_ckpt_10000 -- so each run's metrics land in their own distinct,
    # non-overwriting namespace instead of all sharing "eval/...".
    metric_prefix: str = "eval"

    #################################################################################################################
    # Value-guided denoising / test-time adaptation. Only takes effect for checkpoints whose
    # config has a guidance_schedule; see run_libero_eval_vectorized's docstring for the flow.
    #################################################################################################################
    # False forces plain unguided eval even on a guidance-capable checkpoint, for A/B comparison.
    enable_guided_eval: bool = True
    num_tta_episodes: int = 10  # unguided probe rollouts per task, used to estimate z_r
    tta_sample_stride: int = 10  # keep every Nth post-warmup (element, reward) pair from probes
    # replay each reserved probe initial state this many times, so the probe
    # budget can grow past the 10 states left over after the 40 eval trials. See the probe loop.
    tta_repeats: int = 1
    max_tta_samples: int = 64  # cap on total (element, reward) pairs fed into test_time_adapt
    # Override the checkpoint's own guidance_schedule.base_weight at eval time (e.g. for a
    # guidance-scale sweep) without needing a separate named TrainConfig per scale value. Only
    # valid when the checkpoint's config has a guidance_schedule field; None = use the value baked
    # into the checkpoint's own training config, unchanged.
    guidance_base_weight: float | None = None
    # the rest of the guidance surface, previously unreachable from the CLI.
    # "constant" (the default everywhere) applies full weight at every denoising step; the anneal
    # shapes fade guidance in as t -> 0, which is what rep_grad_guidance.py's TODO recommends,
    # since x_hat0 is a one-step extrapolation and least trustworthy at large t.
    guidance_schedule_shape: str | None = None  # constant | linear_anneal | power_anneal
    guidance_power: float | None = None  # only used by power_anneal
    # Per-example L2 clip on the guidance gradient. The frozen config bakes in 100_000.0, which
    # never binds; worth sweeping once the gradient is nonzero.
    guidance_grad_clip_norm: float | None = None
    # ablation -- evaluate grad_a Q at the noisy iterate x_t rather than the
    # one-step clean-action estimate x_hat0. Tests whether x_hat0's mode-averaging is costing us
    # anything, at the price of querying F outside the clean-action support it was trained on.
    guidance_eval_at_xt: bool = False
    # ridge for the WHITENED z_r estimator. FB's zero-shot embedding is
    # really z_r = E[psi psi^T]^-1 . E[r psi], which reduces to the plain E[r psi] only when
    # E[psi psi^T] = I -- a condition measured at 0.50-0.70 rather than 0 on every checkpoint
    # here. None = the original (unwhitened) estimator. Try ~0.01-1.0; the ridge is scaled by
    # the covariance's own trace, so one value transfers across checkpoints.
    zr_whiten_ridge: float | None = None
    # subtract the mean reward before projecting onto psi. Equivalent to the
    # contrastive direction E_{r=1}[psi] - E_{r=0}[psi], so every zero-reward probe frame
    # contributes as a baseline instead of being multiplied by zero. See guided_policy's
    # test_time_adapt for the full argument.
    center_reward: bool = False
    # dense reward shaping by time-to-success (see _shape_probe_rewards).
    # None keeps LIBERO's native one-frame sparse reward.
    dense_reward_gamma: float | None = None
    # re-seed the policy RNG per episode batch and per TTA probe phase.
    # Default True: without it, action noise depends on the cumulative infer count, so one
    # episode ending a step early re-randomizes every later episode and two identical runs
    # disagree on ~14.5% of episodes -- more than the guidance treatment moves. Set False to
    # reproduce pre-2026-07-30 runs.
    per_episode_rng: bool = True
    # Gaussian noise added to the base BC velocity at every denoising step,
    # for the positive-control experiment -- degrade a near-perfect policy, then ask whether
    # guidance steers it back. Deliberately placed on v_bc rather than on the final action:
    # guidance IS a correction to v_bc, so noise there is the kind of error guidance could
    # cancel, whereas perturbing the output would sit downstream of guidance and guarantee a
    # null by construction. 0.0 is bit-identical to no degradation.
    bc_velocity_noise: float = 0.0
    bc_velocity_bias: float = 0.0
    num_denoise_steps: int | None = None

    # path to write the structured per-episode/per-task results JSON (see
    # eval_lib.write_results_json). None = don't write one, so this is inert unless asked for.
    # Everything the comparison and statistics tooling needs lives in that file; wandb is for
    # eyeballing a single run, not for comparing many.
    results_json: str | None = None


def main(args: Args) -> dict:
    # Sanity-check spawn-based subprocess env creation BEFORE any JAX/CUDA init in this process
    # (see module docstring point 2) -- a throwaway single real-task env, immediately closed.
    first_suite = args.task_suites[0]
    first_task_id = args.task_ids[0] if args.task_ids else 0
    sanity_task = benchmark.get_benchmark_dict()[first_suite]().get_task(first_task_id)
    sanity_env = SubprocVectorEnv([_make_env_factory(sanity_task, LIBERO_ENV_RESOLUTION, args.seed)])
    sanity_env.reset()
    sanity_env.close()

    # Imports requiring the full JAX/openpi stack are deferred so that this module (and the
    # sanity check above) can run before JAX/CUDA ever initializes in this process.
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config(args.checkpoint.config)

    # generalized from base_weight-only to every guidance knob. The
    # annealing shapes have been implemented in GuidanceWeightSchedule all along but no config
    # uses them and there was no way to reach them from the CLI, so they had never been tried --
    # even though rep_grad_guidance.py's own TODO argues for exactly that (x_hat0 is a one-step
    # extrapolation and is least reliable at large t, so constant-weight guidance trusts it most
    # where it is worst). guidance_grad_clip_norm was likewise unreachable.
    schedule_overrides = {
        k: v
        for k, v in (
            ("base_weight", args.guidance_base_weight),
            ("shape", args.guidance_schedule_shape),
            ("power", args.guidance_power),
        )
        if v is not None
    }
    model_overrides = {}
    if args.guidance_grad_clip_norm is not None:
        model_overrides["guidance_grad_clip_norm"] = args.guidance_grad_clip_norm
    if args.guidance_eval_at_xt:
        if not hasattr(train_config.model, "guidance_eval_at_xt"):
            raise ValueError(
                f"--args.guidance-eval-at-xt was set but {args.checkpoint.config!r}'s model config "
                "has no guidance_eval_at_xt field (only Pi0FBConfig defines it)."
            )
        model_overrides["guidance_eval_at_xt"] = True

    if schedule_overrides or model_overrides:
        if not hasattr(train_config.model, "guidance_schedule"):
            raise ValueError(
                f"a --args.guidance-* override was set but {args.checkpoint.config!r}'s model "
                "config has no guidance_schedule field (not a value-guidance model)."
            )
        if schedule_overrides:
            model_overrides["guidance_schedule"] = dataclasses.replace(
                train_config.model.guidance_schedule, **schedule_overrides
            )
        train_config = dataclasses.replace(
            train_config, model=dataclasses.replace(train_config.model, **model_overrides)
        )
        logging.info(f"Overriding guidance settings -> {schedule_overrides} {model_overrides.keys()}")

    policy_cls = _policy.Policy
    if args.enable_guided_eval and hasattr(train_config.model, "guidance_schedule"):
        from openpi.policies import guided_policy as _guided_policy

        policy_cls = _guided_policy.DenoisingGuidedPolicy
        logging.info(
            f"Checkpoint config {args.checkpoint.config!r} supports guided eval -- using DenoisingGuidedPolicy."
        )
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint.dir, policy_cls=policy_cls)

    wandb_run_id = args.wandb_run_id
    if wandb_run_id is None and args.wandb_enabled and args.resume_wandb_run:
        run_dir = pathlib.Path(args.checkpoint.dir).parent
        wandb_id_file = run_dir / "wandb_id.txt"
        if wandb_id_file.exists():
            wandb_run_id = wandb_id_file.read_text().strip()
            logging.info(f"Resuming wandb run {wandb_run_id} from {wandb_id_file}")
        else:
            logging.warning(f"No wandb_id.txt found in {run_dir}; starting a fresh standalone wandb run instead.")
    elif wandb_run_id is None and args.wandb_enabled:
        logging.info("resume_wandb_run=False -- starting a fresh, standalone wandb run.")

    # `main` now returns the metrics dict so a Slurm payload can import this
    # module and call it directly instead of scraping stdout. tyro.cli ignores return values, so
    # the CLI behaves exactly as before.
    return run_libero_eval_vectorized(
        policy,
        task_suites=args.task_suites,
        task_ids=args.task_ids,
        num_trials_per_task=args.num_trials_per_task,
        num_parallel_envs=args.num_parallel_envs,
        resize_size=args.resize_size,
        replan_steps=args.replan_steps,
        num_steps_wait=args.num_steps_wait,
        seed=args.seed,
        video_out_dir=args.video_out_dir,
        log_videos_to_wandb=args.log_videos_to_wandb,
        max_videos_per_task=args.max_videos_per_task,
        wandb_enabled=args.wandb_enabled,
        wandb_run_id=wandb_run_id,
        wandb_project=args.wandb_project or train_config.project_name,
        enable_guided_eval=args.enable_guided_eval,
        num_tta_episodes=args.num_tta_episodes,
        tta_sample_stride=args.tta_sample_stride,
        tta_repeats=args.tta_repeats,
        max_tta_samples=args.max_tta_samples,
        zr_whiten_ridge=args.zr_whiten_ridge,
        center_reward=args.center_reward,
        dense_reward_gamma=args.dense_reward_gamma,
        bc_velocity_noise=args.bc_velocity_noise,
        bc_velocity_bias=args.bc_velocity_bias,
        num_denoise_steps=args.num_denoise_steps,
        per_episode_rng=args.per_episode_rng,
        metric_prefix=args.metric_prefix,
        results_json=args.results_json,
        # The full Args are recorded verbatim in the JSON: it is the reproduction record, and
        # write_results_json derives the pairing fingerprint from a subset of these fields.
        run_args=dataclasses.asdict(args),
    )


if __name__ == "__main__":
    # force=True: _resolve_mujoco_egl_device_id() (called at module import time, above) imports
    # mujoco.egl directly in this (parent) process -- something in that import chain configures
    # the root logger as a side effect, making a plain basicConfig() here a silent no-op (it only
    # takes effect if the root logger has no handlers yet) and suppressing all logging.info/
    # warning output for the rest of the script. force=True makes this call always win.
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
