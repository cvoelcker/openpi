"""Evaluate a trained policy on LIBERO benchmark suites, with wandb logging attached to the
SAME run that produced the checkpoint.

Mirrors main.py's rollout logic (env setup, image preprocessing, action-chunk replanning, success
detection, video saving) but is built around a `BasePolicy`-conforming policy object rather than a
`WebsocketClientPolicy`, so it can be driven two ways that produce identical evaluations:

  1. In-process from a checkpoint, via `policy_config.create_trained_policy(...)`.
  2. Programmatically, by passing any `BasePolicy` into `run_libero_eval` -- an in-process
     `Policy`, a `WebsocketClientPolicy`, or a `PolicyRecorder` wrapping either.

Unlike main.py, which keeps LIBERO/robosuite/mujoco in their own venv behind a websocket, path (1)
imports the full JAX/openpi stack, so it needs both installed together. Run it as
`uv run --group libero scripts/eval_on_libero.py ...` (see pyproject.toml's `libero` group). To
keep the stacks decoupled, use main.py's websocket path and pass the resulting client policy in.

Usage:
uv run --group libero scripts/eval_on_libero.py \
    --args.checkpoint.config=pi05_libero \
    --args.checkpoint.dir=checkpoints/pi05_libero/my_experiment/20000 \
    --args.task-suites libero_spatial libero_object \
    --args.num-trials-per-task=10
"""

import collections
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import platform
import socket
import subprocess
from typing import Protocol

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tqdm
import tyro
import wandb

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Longest training-demo step count per suite, +margin -- same values as main.py.
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class _InferPolicy(Protocol):
    """The only interface this script relies on -- matches openpi_client.base_policy.BasePolicy
    (in-process Policy, WebsocketClientPolicy, and PolicyRecorder all satisfy this already), kept
    as a local Protocol so this file doesn't need to import openpi_client at type-check time for
    the parts of it that don't otherwise need it."""

    def infer(self, obs: dict) -> dict: ...


@dataclasses.dataclass(frozen=True)
class CheckpointSource:
    """Build a policy from a trained checkpoint (mirrors scripts/serve_policy.py's `Checkpoint`)."""

    config: str  # Training config name (e.g. "pi05_libero")
    dir: str  # Checkpoint directory (e.g. "checkpoints/pi05_libero/my_experiment/20000")


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
    suite: str
    task_id: int
    task_description: str
    episode_idx: int
    success: bool
    length: int
    video_path: pathlib.Path | None
    # Per-episode mean of the guided-denoising diagnostics collected from this episode's
    # infer() calls (see _run_episode), or None for unguided/TTA-probe episodes.
    guidance_summary: dict | None = None
    # The three fields below let results from separate runs be compared with a PAIRED test:
    # recording the state index and seed lets the aggregator VERIFY the pairing rather than assume
    # it. `env_seed` depends on num_parallel_envs, so changing that flag between cells silently
    # re-seeds every episode and would invalidate the comparison.
    initial_state_index: int | None = None
    env_seed: int | None = None
    # None on success. Set when the episode aborted on an exception (EGL/mujoco failures are
    # the common case under parallel envs). Distinguishing this from an honest task failure
    # matters: without it a crashed batch is indistinguishable from "this condition scored 0%".
    error: str | None = None


def _get_libero_env(task, resolution: int, seed: int):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite (see main.py for the same helper)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _obs_to_element(obs, task_description: str, resize_size: int) -> tuple[dict, np.ndarray]:
    """Preprocesses a LIBERO env observation into the dict a Policy.infer/WebsocketClientPolicy.
    infer call expects, matching main.py's preprocessing exactly. Returns (element, display_img)
    where display_img is the (already resized) third-person view, kept separately for the replay
    video.
    """
    # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))
    element = {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": str(task_description),
    }
    return element, img


def _run_episode(
    policy: _InferPolicy,
    env,
    task_description: str,
    initial_state,
    *,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    step_pbar: tqdm.tqdm | None = None,
    infer_fn=None,
) -> tuple[bool, int, list[np.ndarray], list[dict]]:
    """Runs one episode with a fixed initial state. Returns (success, steps_taken, replay_images,
    episode_diagnostics). Mirrors main.py's step loop (action-chunk replanning, wait-for-objects-
    to-settle warmup).

    `step_pbar`, if given, is updated once per env step (including the wait-phase steps) --
    caller owns its lifetime (reset/description/close), this function only calls `.update(1)`.

    `infer_fn`, if given, replaces `policy.infer` as the action-chunk source (e.g.
    `policy.infer_unguided` for TTA probe rollouts, or a guided policy's own `infer` once z_r has
    been set) -- defaults to `policy.infer`, fully backward compatible.

    `episode_diagnostics` collects the `out["guidance_diagnostics"]` dict from every `infer()` call
    that produces one (guided policies only -- see guided_policy.py's `infer`); empty for plain
    policies or unguided calls.
    """
    infer = infer_fn if infer_fn is not None else policy.infer
    env.reset()
    action_plan: collections.deque = collections.deque()
    obs = env.set_init_state(initial_state)
    replay_images: list[np.ndarray] = []
    episode_diagnostics: list[dict] = []

    t = 0
    steps_taken = 0
    success = False
    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                if step_pbar is not None:
                    step_pbar.update(1)
                continue

            element, display_img = _obs_to_element(obs, task_description, resize_size)
            replay_images.append(display_img)

            if not action_plan:
                out = infer(element)
                action_chunk = out["actions"]
                assert len(action_chunk) >= replan_steps, (
                    f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                )
                action_plan.extend(action_chunk[:replan_steps])
                if "guidance_diagnostics" in out:
                    episode_diagnostics.append(out["guidance_diagnostics"])

            action = action_plan.popleft()
            obs, _, done, _ = env.step(action.tolist())
            steps_taken += 1
            t += 1
            if step_pbar is not None:
                step_pbar.update(1)
            if done:
                success = True
                break
    except Exception:
        logging.exception(f"Episode failed for task {task_description!r}")

    return success, steps_taken, replay_images, episode_diagnostics


def _run_tta_probe_episode(
    policy: _InferPolicy,
    env,
    task_description: str,
    initial_state,
    *,
    resize_size: int,
    replan_steps: int,
    num_steps_wait: int,
    max_steps: int,
    tta_sample_stride: int,
    step_pbar: tqdm.tqdm | None = None,
) -> list[tuple[dict, float]]:
    """Runs one UNGUIDED probe episode (via `policy.infer_unguided`, since z_r doesn't exist yet),
    collecting (element, reward) pairs from the env's own native `env.step` reward -- LIBERO/
    robosuite tasks are sparse-reward (1.0 at goal-completion, 0.0 otherwise), so this needs no
    synthetic reward heuristic. Kept for building a `RewardSamples` (see `test_time_adapt`), not
    for success/video tracking, so it has no `done`-triggered success flag of its own beyond
    always keeping the terminal step's sample.

    Samples every `tta_sample_stride`-th post-warmup step, AND always the terminal/`done` step (so
    a late sparse success isn't missed by striding). Exceptions are caught and logged (mirrors
    `_run_episode`), returning whatever was collected so far.
    """
    env.reset()
    action_plan: collections.deque = collections.deque()
    obs = env.set_init_state(initial_state)
    collected: list[tuple[dict, float]] = []

    t = 0
    step_count = 0
    try:
        while t < max_steps + num_steps_wait:
            if t < num_steps_wait:
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                if step_pbar is not None:
                    step_pbar.update(1)
                continue

            element, _ = _obs_to_element(obs, task_description, resize_size)

            if not action_plan:
                action_chunk = policy.infer_unguided(element)["actions"]
                action_plan.extend(action_chunk[:replan_steps])

            action = action_plan.popleft()
            obs, reward, done, _ = env.step(action.tolist())
            step_count += 1
            t += 1
            if step_pbar is not None:
                step_pbar.update(1)

            if step_count % tta_sample_stride == 0 or done:
                collected.append((element, float(reward)))
            if done:
                break
    except Exception:
        logging.exception(f"TTA probe episode failed for task {task_description!r}")

    return collected


def _summarize_episode_diagnostics(episode_diagnostics: list[dict]) -> dict | None:
    """Reduces one episode's list of per-infer-call guidance diagnostics dicts (each holding the
    `*_per_step` arrays plus `q_value_delta`/`x0_norm` scalars produced by
    `sample_actions_guided(..., return_diagnostics=True)`, see guided_policy.py's `infer`) into a
    single flat dict of episode-level means, for `EpisodeResult.guidance_summary` and the
    per-episode wandb.Table in `_aggregate_and_log`. Returns None for episodes that collected no
    guidance diagnostics (unguided policy, or an unguided call within a guided eval)."""
    if not episode_diagnostics:
        return None
    return {
        "mean_grad_norm": float(np.mean([np.mean(d["guidance_grad_norm_per_step"]) for d in episode_diagnostics])),
        "mean_bc_velocity_norm": float(np.mean([np.mean(d["bc_velocity_norm_per_step"]) for d in episode_diagnostics])),
        "mean_alignment": float(np.mean([np.mean(d["guidance_alignment_per_step"]) for d in episode_diagnostics])),
        "clip_hit_frac": float(np.mean([np.mean(d["guidance_clip_hit_frac_per_step"]) for d in episode_diagnostics])),
        "mean_relative_guidance_magnitude": float(
            np.mean([np.mean(d["relative_guidance_magnitude_per_step"]) for d in episode_diagnostics])
        ),
        "mean_q_value": float(np.mean([np.mean(d["q_value_per_step"]) for d in episode_diagnostics])),
        "mean_q_value_delta": float(np.mean([d["q_value_delta"] for d in episode_diagnostics])),
        "num_inference_calls": len(episode_diagnostics),
    }


# cap on the per-denoising-step curves kept in the results JSON. The
# curves are indexed by denoising step (10 by default), so this only ever bites if num_steps is
# raised a long way; it exists so the JSON can never grow with episode count or episode length.
MAX_CURVE_STEPS = 64


def _log_task_guidance_diagnostics(
    suite_name: str, task_id: int, task_diagnostics: list[dict], *, wandb_enabled: bool, metric_prefix: str = "eval"
) -> dict | None:
    """Logs per-task guidance scalar summary keys plus a `denoising_trajectory` wandb.Table (one
    row per denoising step index, averaged over every infer() call across every guided episode
    run for this task). `task_diagnostics` is the flat concatenation of every episode's
    `episode_diagnostics` list for this task -- no-op if empty (unguided task).

    `metric_prefix` replaces the leading "eval" in every logged key (e.g. "eval_ckpt_10k") -- see
    `run_libero_eval`'s docstring for why.

    Returns `{"scalars": ..., "curves": ...}` (None for an unguided task) so callers can persist
    these offline, rather than only dropping them into wandb.
    """
    if not task_diagnostics:
        return None

    def _mean_over_calls(key: str) -> float:
        return float(np.mean([np.mean(d[key]) for d in task_diagnostics]))

    prefix = f"{metric_prefix}/guidance/{suite_name}/task{task_id}"
    scalar_metrics = {
        f"{prefix}/mean_grad_norm": _mean_over_calls("guidance_grad_norm_per_step"),
        f"{prefix}/mean_bc_velocity_norm": _mean_over_calls("bc_velocity_norm_per_step"),
        f"{prefix}/mean_alignment": _mean_over_calls("guidance_alignment_per_step"),
        f"{prefix}/clip_hit_frac": _mean_over_calls("guidance_clip_hit_frac_per_step"),
        f"{prefix}/mean_relative_guidance_magnitude": _mean_over_calls("relative_guidance_magnitude_per_step"),
        f"{prefix}/mean_q_value": _mean_over_calls("q_value_per_step"),
        f"{prefix}/mean_q_value_delta": float(np.mean([d["q_value_delta"] for d in task_diagnostics])),
        f"{prefix}/num_inference_calls": len(task_diagnostics),
    }
    # The fixed-bias positive control's readout, guarded by presence because sample_actions_guided
    # only emits it on paths that compute it. This aggregator is an explicit allowlist: anything
    # not listed here is silently dropped, which cost one whole run of the positive control.
    if "bc_bias_pushback_per_step" in task_diagnostics[0]:
        scalar_metrics[f"{prefix}/bc_bias_pushback_cos"] = _mean_over_calls("bc_bias_pushback_per_step")
    logging.info(
        f"[{suite_name}#{task_id}] guidance: mean_alignment={scalar_metrics[f'{prefix}/mean_alignment']:.3f}, "
        f"clip_hit_frac={scalar_metrics[f'{prefix}/clip_hit_frac']:.3f}, "
        f"mean_q_value_delta={scalar_metrics[f'{prefix}/mean_q_value_delta']:.4f}"
        + (
            f", bc_bias_pushback_cos={scalar_metrics[f'{prefix}/bc_bias_pushback_cos']:.4f}"
            if f"{prefix}/bc_bias_pushback_cos" in scalar_metrics
            else ""
        )
    )

    # build the per-step curves unconditionally (they used to live inside
    # the wandb branch), so they are returned and persisted even with wandb disabled. The wandb
    # table is now just a second consumer of the same arrays.
    num_steps = min(len(task_diagnostics[0]["guidance_weight_per_step"]), MAX_CURVE_STEPS)
    curve_keys = (
        "guidance_grad_norm_per_step",
        "bc_velocity_norm_per_step",
        "guidance_alignment_per_step",
        "guidance_clip_hit_frac_per_step",
        "relative_guidance_magnitude_per_step",
        "q_value_per_step",
        "x_hat0_norm_per_step",
        "guidance_weight_per_step",
    )
    curves = {
        key: [float(np.mean([d[key][step] for d in task_diagnostics])) for step in range(num_steps)]
        for key in curve_keys
    }

    if wandb_enabled:
        table = wandb.Table(
            columns=[
                "step",
                "mean_grad_norm",
                "mean_bc_velocity_norm",
                "mean_alignment",
                "clip_hit_frac",
                "mean_relative_guidance_magnitude",
                "mean_q_value",
                "mean_x_hat0_norm",
            ]
        )
        for step in range(num_steps):
            table.add_data(
                step,
                curves["guidance_grad_norm_per_step"][step],
                curves["bc_velocity_norm_per_step"][step],
                curves["guidance_alignment_per_step"][step],
                curves["guidance_clip_hit_frac_per_step"][step],
                curves["relative_guidance_magnitude_per_step"][step],
                curves["q_value_per_step"][step],
                curves["x_hat0_norm_per_step"][step],
            )
        wandb.log({**scalar_metrics, f"{prefix}/denoising_trajectory": table})

    # Strip the wandb key path off the scalar names -- inside the JSON the suite/task/prefix are
    # already structural, so `mean_grad_norm` is the useful key, not the full slash path.
    return {
        "scalars": {k.rsplit("/", 1)[-1]: v for k, v in scalar_metrics.items()},
        "curves": curves,
    }


def _aggregate_and_log(results: list[EpisodeResult], *, wandb_enabled: bool, metric_prefix: str = "eval") -> dict:
    """Computes per-task success rate, per-suite average, overall average, and episode-length
    stats, and (if enabled) logs a per-task wandb.Table plus scalar summary metrics.

    `metric_prefix` replaces the leading "eval" in every logged key -- see `run_libero_eval`'s
    docstring for why."""
    by_task: dict[tuple[str, int], list[EpisodeResult]] = collections.defaultdict(list)
    by_suite: dict[str, list[EpisodeResult]] = collections.defaultdict(list)
    for r in results:
        by_task[(r.suite, r.task_id)].append(r)
        by_suite[r.suite].append(r)

    per_task_rows = []
    for (suite, task_id), episodes in sorted(by_task.items()):
        n = len(episodes)
        successes = sum(e.success for e in episodes)
        mean_len = float(np.mean([e.length for e in episodes]))
        per_task_rows.append(
            {
                "suite": suite,
                "task_id": task_id,
                "task_description": episodes[0].task_description,
                "num_episodes": n,
                "success_rate": successes / n,
                "mean_episode_length": mean_len,
            }
        )

    per_suite_metrics = {}
    for suite, episodes in sorted(by_suite.items()):
        n = len(episodes)
        successes = sum(e.success for e in episodes)
        per_suite_metrics[f"{metric_prefix}/{suite}/success_rate"] = successes / n
        per_suite_metrics[f"{metric_prefix}/{suite}/mean_episode_length"] = float(np.mean([e.length for e in episodes]))
        per_suite_metrics[f"{metric_prefix}/{suite}/num_episodes"] = n

    overall_n = len(results)
    overall_successes = sum(r.success for r in results)
    overall_metrics = {
        f"{metric_prefix}/overall/success_rate": overall_successes / overall_n if overall_n else float("nan"),
        f"{metric_prefix}/overall/mean_episode_length": float(np.mean([r.length for r in results]))
        if overall_n
        else float("nan"),
        f"{metric_prefix}/overall/num_episodes": overall_n,
        # Average of per-suite success rates (as opposed to the pooled overall rate above, which
        # implicitly weights by how many episodes each suite ran) -- both are useful, since suites
        # can have very different task/trial counts.
        f"{metric_prefix}/average_suite_success_rate": float(
            np.mean([m for k, m in per_suite_metrics.items() if k.endswith("/success_rate")])
        )
        if per_suite_metrics
        else float("nan"),
    }

    for row in per_task_rows:
        logging.info(
            f"[{row['suite']}] task {row['task_id']} ({row['task_description']!r}): "
            f"{row['success_rate']:.2%} success over {row['num_episodes']} episodes, "
            f"mean length {row['mean_episode_length']:.1f}"
        )
    for suite, n in ((s, len(e)) for s, e in by_suite.items()):
        logging.info(
            f"[{suite}] success_rate={per_suite_metrics[f'{metric_prefix}/{suite}/success_rate']:.2%} over {n} episodes"
        )
    logging.info(
        f"OVERALL success_rate={overall_metrics[f'{metric_prefix}/overall/success_rate']:.2%} over {overall_n} episodes"
    )

    if wandb_enabled:
        table = wandb.Table(
            columns=["suite", "task_id", "task_description", "num_episodes", "success_rate", "mean_episode_length"]
        )
        for row in per_task_rows:
            table.add_data(
                row["suite"],
                row["task_id"],
                row["task_description"],
                row["num_episodes"],
                row["success_rate"],
                row["mean_episode_length"],
            )
        log_payload = {f"{metric_prefix}/per_task_results": table, **per_suite_metrics, **overall_metrics}

        guided_results = [r for r in results if r.guidance_summary is not None]
        if guided_results:
            per_episode_table = wandb.Table(
                columns=[
                    "suite",
                    "task_id",
                    "success",
                    "episode_length",
                    "mean_grad_norm",
                    "mean_alignment",
                    "clip_hit_frac",
                    "mean_q_value_delta",
                    "z_r_norm",
                ]
            )
            for r in guided_results:
                gs = r.guidance_summary
                per_episode_table.add_data(
                    r.suite,
                    r.task_id,
                    r.success,
                    r.length,
                    gs["mean_grad_norm"],
                    gs["mean_alignment"],
                    gs["clip_hit_frac"],
                    gs["mean_q_value_delta"],
                    gs.get("z_r_norm", float("nan")),
                )
            log_payload[f"{metric_prefix}/guidance/per_episode_results"] = per_episode_table

        wandb.log(log_payload)

    return {"per_task": per_task_rows, "per_suite": per_suite_metrics, "overall": overall_metrics}


# everything below (through write_results_json) exists because these
# scripts previously left no machine-readable trace -- results went to wandb and to log lines,
# so comparing two runs meant scraping stdout. Structured output is what makes a guidance-scale
# sweep analyzable (and statistically testable) after the fact.

# Args fields that must be identical for two cells to be a valid paired comparison. seed and
# num_parallel_envs jointly determine each episode's env seed (seed + episode_idx % b_task), and
# the rest determine what an "episode" even is; if any differ, the cells are not matched samples.
PAIRING_FIELDS = (
    "seed",
    "num_parallel_envs",
    "num_trials_per_task",
    "num_steps_wait",
    "replan_steps",
    "resize_size",
    "num_tta_episodes",
    "tta_sample_stride",
    "max_tta_samples",
    # changes the action-noise stream, so runs with and without it are
    # not matched samples.
    "per_episode_rng",
    # the degradation level defines the experiment; arms at different
    # sigma are not matched samples.
    "bc_velocity_noise",
    "bc_velocity_bias",
    "num_denoise_steps",
)


def subsample_probe_samples(probe_samples: list[tuple[dict, float]], max_samples: int) -> list[tuple[dict, float]]:
    """Cap TTA probe samples at `max_samples`, keeping every nonzero-reward one.

    LIBERO's reward is sparse -- 1.0 only on the success step -- so of ~400 samples per task
    perhaps 5 are nonzero, and a flat uniform subsample to 64 keeps ~16% of those. But

        z_r = mean_i(r_i * psi(s_i)) = (1/N) * sum over the nonzero-reward samples

    depends on the zero-reward samples only through N, which `normalize_z_r` divides out. So keep
    every nonzero sample and spend the remaining budget on a uniform subsample of the zeros
    (retained so an unnormalized z_r keeps a sane scale).
    """
    if len(probe_samples) <= max_samples:
        return probe_samples
    nonzero = [s for s in probe_samples if s[1] > 0]
    zeros = [s for s in probe_samples if s[1] <= 0]
    budget = max(0, max_samples - len(nonzero))
    if budget and zeros:
        idxs = np.linspace(0, len(zeros) - 1, min(budget, len(zeros))).astype(int)
        zeros = [zeros[i] for i in idxs]
    else:
        zeros = []
    if len(nonzero) > max_samples:
        # More successes than the cap: subsample them uniformly rather than truncating, so the
        # retained set still spans the whole probe phase.
        idxs = np.linspace(0, len(nonzero) - 1, max_samples).astype(int)
        return [nonzero[i] for i in idxs]
    return nonzero + zeros


def _provenance() -> dict:
    """Git + Slurm + host identity for the results file. Lives here rather than in the launcher
    so an interactive run records the same fields as a submitted one."""

    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
        "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
    }


def _jsonable(obj):
    """numpy/pathlib -> plain JSON types. Diagnostics arrive as numpy scalars and arrays."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.floating | np.integer):
        item = obj.item()
        return None if isinstance(item, float) and not math.isfinite(item) else item
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        # JSON has no NaN/Infinity; json.dump would emit bare `NaN`, which is invalid JSON and
        # breaks strict parsers. Diagnostics legitimately contain NaN (e.g. TTA pairwise
        # consistency with <2 nonzero-reward samples), so map them to null.
        return obj if math.isfinite(obj) else None
    return obj


def write_results_json(
    path: str | pathlib.Path,
    *,
    results: list[EpisodeResult],
    metrics: dict,
    run_args: dict,
    tta_by_task: dict | None = None,
    guidance_by_task: dict | None = None,
    extra: dict | None = None,
) -> pathlib.Path:
    """Serializes one eval cell to JSON. `metrics` is exactly what `_aggregate_and_log` returns;
    `tta_by_task`/`guidance_by_task` are keyed by (suite, task_id).

    Written via a temp file in the same directory plus os.replace, so a reader can never observe
    a half-written file -- an aggregator polling a sweep would otherwise occasionally parse a
    truncated cell as a complete one.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tta_by_task = tta_by_task or {}
    guidance_by_task = guidance_by_task or {}

    errored = [r for r in results if r.error is not None]
    per_task = [
        {
            **row,
            "tta": _jsonable(tta_by_task.get((row["suite"], row["task_id"]))),
            "guidance": _jsonable(guidance_by_task.get((row["suite"], row["task_id"]))),
        }
        for row in metrics["per_task"]
    ]

    payload = {
        "schema_version": 1,
        # "ok" only when every episode completed. A cell with any errored episode must not be
        # silently averaged into a comparison -- see EpisodeResult.error.
        "status": "ok" if not errored else "partial",
        "num_episodes_errored": len(errored),
        "provenance": {**_provenance(), "finished_utc": datetime.datetime.now(datetime.UTC).isoformat()},
        "args": _jsonable(run_args),
        "pairing_fields": list(PAIRING_FIELDS),
        "pairing_fingerprint": hashlib.sha256(
            json.dumps({k: run_args.get(k) for k in PAIRING_FIELDS}, sort_keys=True).encode()
        ).hexdigest(),
        "episodes": [
            {
                "suite": r.suite,
                "task_id": r.task_id,
                "task_description": r.task_description,
                "episode_idx": r.episode_idx,
                "initial_state_index": r.initial_state_index,
                "env_seed": r.env_seed,
                "success": bool(r.success),
                "length": int(r.length),
                "error": r.error,
                "video": str(r.video_path) if r.video_path else None,
                "guidance": _jsonable(r.guidance_summary),
            }
            for r in results
        ],
        "per_task": per_task,
        "per_suite": _jsonable(metrics["per_suite"]),
        "overall": _jsonable(metrics["overall"]),
        **(extra or {}),
    }

    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=1, allow_nan=False)
    os.replace(tmp, path)
    logging.info(f"Wrote results JSON -> {path} ({len(results)} episodes, status={payload['status']})")
    return path


def _save_episode_video(
    replay_images: list[np.ndarray],
    *,
    suite_name: str,
    task_id: int,
    task_description: str,
    episode_idx: int,
    success: bool,
    video_out_dir: pathlib.Path,
    wandb_enabled: bool,
    log_videos_to_wandb: bool,
    max_videos_per_task: int,
    metric_prefix: str = "eval",
) -> pathlib.Path | None:
    """Writes `replay_images` to an .mp4 (if non-empty) and optionally logs it to wandb. Shared
    between the sequential and vectorized eval scripts so video-naming/wandb-logging conventions
    can never silently drift between them.

    `metric_prefix` replaces the leading "eval" in the logged key -- see `run_libero_eval`'s
    docstring for why."""
    if not replay_images:
        return None
    suffix = "success" if success else "failure"
    task_segment = task_description.replace(" ", "_")
    fname = f"{suite_name}_task{task_id}_{task_segment}_ep{episode_idx}_{suffix}.mp4"
    video_path = video_out_dir / fname
    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
    if wandb_enabled and log_videos_to_wandb and episode_idx < max_videos_per_task:
        wandb.log(
            {
                f"{metric_prefix}/videos/{suite_name}/task{task_id}_{suffix}": wandb.Video(
                    str(video_path), fps=10, format="mp4"
                )
            }
        )
    return video_path


def run_libero_eval(
    policy: _InferPolicy,
    *,
    task_suites: tuple[str, ...] = ("libero_spatial",),
    task_ids: dict[str, tuple[int, ...] | None] | tuple[int, ...] | None = None,
    num_trials_per_task: int = 50,
    resize_size: int = 224,
    replan_steps: int = 5,
    num_steps_wait: int = 10,
    seed: int = 7,
    video_out_dir: str | pathlib.Path = "data/libero/videos",
    log_videos_to_wandb: bool = True,
    max_videos_per_task: int = 1,
    wandb_enabled: bool = True,
    wandb_run_id: str | None = None,
    wandb_project: str | None = None,
    enable_guided_eval: bool = True,
    num_tta_episodes: int = 3,
    tta_sample_stride: int = 10,
    max_tta_samples: int = 64,
    metric_prefix: str = "eval",
) -> dict:
    """Evaluates `policy` on every task in every requested LIBERO suite, logging per-task/
    per-suite/overall success rates, episode lengths, and rollout videos.

    `metric_prefix` (default "eval") replaces the leading "eval" in every wandb key. Override it
    (e.g. "eval_ckpt_10000") when evaluating several checkpoints against the same resumed wandb
    run, so each lands in its own non-overwriting namespace.

    `task_ids` controls which task indices to run within each suite: `None` (default) runs every
    task in every requested suite; a flat tuple of ints applies the same task-index subset to
    every requested suite (CLI-friendly shorthand); a `{suite_name: ids_or_None}` dict gives
    per-suite control (only available when calling this function directly, not from the CLI).

    wandb: if `wandb_run_id` is given, attaches to that EXISTING run (`resume="must"`) so eval
    metrics land in the same run the checkpoint came from -- see `main()` for how the CLI wrapper
    derives this automatically from a checkpoint directory's `wandb_id.txt`. If `wandb_run_id` is
    None but `wandb_enabled` is True, starts a fresh standalone run instead. Set
    `wandb_enabled=False` to skip wandb entirely (results are still logged via `logging`).

    Two-phase evaluation for value-guided-denoising policies (`DenoisingGuidedPolicy` --
    duck-typed via `hasattr(policy, "test_time_adapt")`, so this is a no-op for plain policies):
    if `enable_guided_eval` and the policy supports it, each task first runs `num_tta_episodes`
    UNGUIDED probe rollouts (using `initial_states[-num_tta_episodes:]`, distinct from the real
    eval episodes' `initial_states[0:num_trials_per_task]`, to avoid contaminating reported
    metrics with states the policy was just "peeking" at), collects up to `max_tta_samples`
    (element, reward) pairs (subsampled uniformly if more were collected), estimates z_r via
    `test_time_adapt`, then runs the real episodes GUIDED by that z_r. Falls back to unguided eval
    for a task (logged, not an error) if TTA collects zero samples or `num_tta_episodes == 0`.
    """
    if wandb_enabled:
        if wandb_run_id is not None:
            wandb.init(id=wandb_run_id, resume="must", project=wandb_project)
        else:
            logging.warning("No wandb_run_id given; starting a fresh standalone wandb run.")
            wandb.init(project=wandb_project or "libero_eval")

        # Guidance-schedule visibility: makes it clear in the (possibly resumed, training-run)
        # wandb config which guidance settings actually produced THIS eval, since eval can
        # override these independently of whatever the checkpoint's own training config had.
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
    benchmark_dict = benchmark.get_benchmark_dict()
    results: list[EpisodeResult] = []

    # Four persistent, fixed-position bars (suite / task / episode / step), reused (via reset())
    # rather than recreated each iteration -- recreating tqdm objects inside nested loops causes
    # scroll/flicker artifacts instead of a stable, continually-ticking nested display.
    suite_pbar = tqdm.tqdm(total=len(task_suites), desc="suites", position=0, leave=True)
    task_pbar = tqdm.tqdm(total=0, desc="tasks", position=1, leave=True)
    episode_pbar = tqdm.tqdm(total=num_trials_per_task, desc="episodes", position=2, leave=True)
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
                initial_states = task_suite.get_task_init_states(task_id)
                env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

                task_pbar.set_description(f"[{suite_name}] task {task_id}: {task_description[:40]}")

                active_infer = policy.infer
                task_z_r_norm = float("nan")
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
                        probe_samples: list[tuple[dict, float]] = []
                        for tta_idx, init_state in enumerate(tta_initial_states):
                            task_pbar.set_description(f"[{suite_name}#{task_id}] TTA probe {tta_idx}")
                            step_pbar.reset(total=max_steps + num_steps_wait)
                            step_pbar.set_description(f"[{suite_name}#{task_id}] TTA probe {tta_idx} steps")
                            probe_samples.extend(
                                _run_tta_probe_episode(
                                    policy,
                                    env,
                                    task_description,
                                    init_state,
                                    resize_size=resize_size,
                                    replan_steps=replan_steps,
                                    num_steps_wait=num_steps_wait,
                                    max_steps=max_steps,
                                    tta_sample_stride=tta_sample_stride,
                                    step_pbar=step_pbar,
                                )
                            )
                        # stratified, so the rare nonzero-reward samples
                        # (the only ones z_r actually depends on) survive the cap.
                        probe_samples = subsample_probe_samples(probe_samples, max_tta_samples)

                        if probe_samples:
                            elements = [e for e, _ in probe_samples]
                            rewards = np.array([r for _, r in probe_samples], dtype=np.float32)
                            reward_samples = policy.build_reward_samples(elements, rewards)
                            z_r = policy.test_time_adapt(reward_samples)
                            z_r_norm = float(np.linalg.norm(z_r))
                            task_z_r_norm = z_r_norm
                            tta_diag = getattr(policy, "_last_tta_diagnostics", {})
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
                            active_infer = policy.infer  # now guided, since z_r is set
                        else:
                            logging.warning(f"[{suite_name}#{task_id}] TTA collected 0 samples; using unguided eval.")
                            active_infer = policy.infer_unguided
                    else:
                        logging.info(
                            f"[{suite_name}#{task_id}] num_tta_episodes=0; guided eval requested "
                            "but no z_r estimated -- using unguided eval for this task."
                        )
                        active_infer = policy.infer_unguided

                episode_pbar.reset(total=num_trials_per_task)
                task_guidance_diagnostics: list[dict] = []

                for episode_idx in range(num_trials_per_task):
                    episode_pbar.set_description(f"[{suite_name}#{task_id}] episode {episode_idx}")
                    step_pbar.reset(total=max_steps + num_steps_wait)
                    step_pbar.set_description(f"[{suite_name}#{task_id} ep{episode_idx}] steps")

                    success, steps_taken, replay_images, episode_diagnostics = _run_episode(
                        policy,
                        env,
                        task_description,
                        initial_states[episode_idx],
                        resize_size=resize_size,
                        replan_steps=replan_steps,
                        num_steps_wait=num_steps_wait,
                        max_steps=max_steps,
                        step_pbar=step_pbar,
                        infer_fn=active_infer,
                    )
                    task_guidance_diagnostics.extend(episode_diagnostics)

                    video_path = _save_episode_video(
                        replay_images,
                        suite_name=suite_name,
                        task_id=task_id,
                        task_description=task_description,
                        episode_idx=episode_idx,
                        success=success,
                        video_out_dir=video_out_dir,
                        wandb_enabled=wandb_enabled,
                        log_videos_to_wandb=log_videos_to_wandb,
                        max_videos_per_task=max_videos_per_task,
                        metric_prefix=metric_prefix,
                    )

                    guidance_summary = _summarize_episode_diagnostics(episode_diagnostics)
                    if guidance_summary is not None:
                        guidance_summary["z_r_norm"] = task_z_r_norm

                    results.append(
                        EpisodeResult(
                            suite_name,
                            task_id,
                            task_description,
                            episode_idx,
                            success,
                            steps_taken,
                            video_path,
                            guidance_summary,
                        )
                    )
                    episode_pbar.update(1)

                _log_task_guidance_diagnostics(
                    suite_name,
                    task_id,
                    task_guidance_diagnostics,
                    wandb_enabled=wandb_enabled,
                    metric_prefix=metric_prefix,
                )
                env.close()
                task_pbar.update(1)

            suite_pbar.update(1)
    finally:
        step_pbar.close()
        episode_pbar.close()
        task_pbar.close()
        suite_pbar.close()

    metrics = _aggregate_and_log(results, wandb_enabled=wandb_enabled, metric_prefix=metric_prefix)
    if wandb_enabled:
        wandb.finish()
    return metrics


@dataclasses.dataclass
class Args:
    # Policy source. Required for CLI use; for programmatic use, call run_libero_eval(...)
    # directly with an already-built policy object instead of going through main().
    checkpoint: CheckpointSource

    #################################################################################################################
    # LIBERO environment-specific parameters (mirrors main.py's Args)
    #################################################################################################################
    task_suites: tuple[str, ...] = (
        "libero_spatial",
    )  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    task_ids: tuple[int, ...] | None = None  # Task indices to run per suite. None = every task in each suite.
    num_trials_per_task: int = 50  # Number of rollouts per task
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    seed: int = 7

    #################################################################################################################
    # Output / logging
    #################################################################################################################
    video_out_dir: str = "data/libero/videos"
    log_videos_to_wandb: bool = True
    max_videos_per_task: int = 3  # cap wandb video uploads per task; local .mp4s are always saved

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
    # config has a guidance_schedule; see run_libero_eval's docstring for the per-task flow.
    #################################################################################################################
    # False forces plain unguided eval even on a guidance-capable checkpoint, for A/B comparison.
    enable_guided_eval: bool = True
    num_tta_episodes: int = 3  # unguided probe rollouts per task, used to estimate z_r
    tta_sample_stride: int = 10  # keep every Nth post-warmup (element, reward) pair from probes
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


def main(args: Args) -> None:
    # Imports requiring the full JAX/openpi stack are deferred into main() so that
    # `run_libero_eval` (and everything above it) can be imported and driven with an
    # externally-constructed policy object in an environment that doesn't have JAX installed
    # (see the module docstring's note on environment separation).
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
        run_dir = pathlib.Path(args.checkpoint.dir).parent  # matches TrainConfig.checkpoint_dir
        wandb_id_file = run_dir / "wandb_id.txt"
        if wandb_id_file.exists():
            wandb_run_id = wandb_id_file.read_text().strip()
            logging.info(f"Resuming wandb run {wandb_run_id} from {wandb_id_file}")
        else:
            logging.warning(f"No wandb_id.txt found in {run_dir}; starting a fresh standalone wandb run instead.")
    elif wandb_run_id is None and args.wandb_enabled:
        logging.info("resume_wandb_run=False -- starting a fresh, standalone wandb run.")

    run_libero_eval(
        policy,
        task_suites=args.task_suites,
        task_ids=args.task_ids,
        num_trials_per_task=args.num_trials_per_task,
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
        max_tta_samples=args.max_tta_samples,
        metric_prefix=args.metric_prefix,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
