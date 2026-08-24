"""Roll out the FROZEN policy on LIBERO and store the result as a LeRobot dataset, failures
included.

    uv run scripts/collect_policy_rollouts.py --config pi05_libero \
        --checkpoint <dir> --suite libero_90 --task-ids 0 1 2 --episodes-per-task 20 \
        --repo-id yajat/libero90_pi05_rollouts --root $SCRATCH/rollouts

The representation heads have only ever been trained on expert demonstrations, where every
trajectory succeeds, so they learn the successor features of the EXPERT occupancy measure. At
deployment they are asked to tell a rollout that will succeed from one that will not -- a
distinction with no variance anywhere in that training data. Measured: Q's within-task partial
correlation with success is +0.043 (p=0.74), and grad_a Q opposes a deliberately injected velocity
bias at cosine +0.046 where a useful correction would be +1.0.

This collects the distribution they need: the frozen policy's own rollouts, including the majority
that fail. On libero_90 the policy is genuinely weak (~26% overall), so failures come for free and
no artificial degradation is needed.

Design notes:

- Norm stats are the POLICY'S own, not the collected data's. Training under stats recomputed from
  these rollouts would reintroduce the train/deploy mismatch that cost 26% -> 16% earlier.
- Feature parity with the expert datasets (image/wrist_image/state/actions, 256x256 uint8, 8-dim
  state, 7-dim action, fps 10), so the transform pipeline treats them identically.
- Actions are stored raw, exactly what was stepped into the env, matching the expert datasets.
- Per-episode outcomes go in a sidecar JSON, not a dataset feature: the loss never reads reward,
  and a feature the expert datasets lack could change transform behaviour for no benefit.
- Images are stored at the env's native 256x256, before the 224 resize applied for the policy --
  the dataset should hold observations, not model-preprocessed tensors.
"""

import dataclasses
import json
import logging

# Must precede any jax/mujoco import -- same prologue as eval_on_libero_vectorized.
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import multiprocessing

import eval_on_libero as eval_lib
import eval_on_libero_vectorized as evec
import numpy as np
import tqdm
import tyro


@dataclasses.dataclass
class Args:
    config: str = "pi05_libero"
    checkpoint: str = ""
    suite: str = "libero_90"
    task_ids: tuple[int, ...] = ()
    episodes_per_task: int = 20
    num_parallel_envs: int = 10
    repo_id: str = "yajat/libero90_pi05_rollouts"
    root: str = ""
    seed: int = 11
    replan_steps: int = 5
    num_steps_wait: int = 10
    resize_size: int = 224  # only for the policy's input; stored frames stay native 256
    fps: int = 10


LIBERO_NATIVE_RES = 256


def _raw_frame(obs) -> dict:
    """The stored observation: native-resolution images with the same 180-degree rotation the
    policy preprocessing applies, plus the 8-dim proprio state. Mirrors
    `eval_on_libero._obs_to_element` but WITHOUT the 224 resize -- see the module docstring."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            eval_lib._quat2axisangle(obs["robot0_eef_quat"]),  # noqa: SLF001 -- intentional reuse
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return {"image": img, "wrist_image": wrist, "state": state}


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from libero.libero import benchmark
    from libero.libero.envs import SubprocVectorEnv

    root = pathlib.Path(args.root).expanduser() if args.root else None
    features = {
        "image": {"dtype": "image", "shape": (LIBERO_NATIVE_RES, LIBERO_NATIVE_RES, 3), "names": ["h", "w", "c"]},
        "wrist_image": {"dtype": "image", "shape": (LIBERO_NATIVE_RES, LIBERO_NATIVE_RES, 3), "names": ["h", "w", "c"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    }
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=root,
        robot_type="panda",
        features=features,
        use_videos=False,  # image dtype, not video: keeps the writer simple and matches v2.0 layout
    )

    # Plain (unguided) policy: we are collecting the frozen policy's own occupancy measure, so
    # nothing here may steer it. Same construction path the eval script uses.
    import openpi.policies.policy as _policy
    import openpi.policies.policy_config as _policy_config
    import openpi.training.config as _config

    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint, policy_cls=_policy.Policy)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = list(args.task_ids) if args.task_ids else list(range(suite.n_tasks))
    max_steps = eval_lib.SUITE_MAX_STEPS[args.suite]
    outcomes: list[dict] = []
    ep_index = 0

    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        b = min(args.num_parallel_envs, args.episodes_per_task)
        env_fns = [evec._make_env_factory(task, evec.LIBERO_ENV_RESOLUTION, args.seed + k) for k in range(b)]  # noqa: SLF001
        venv = SubprocVectorEnv(env_fns)
        try:
            done = 0
            pbar = tqdm.tqdm(total=args.episodes_per_task, desc=f"{args.suite}#{task_id}")
            while done < args.episodes_per_task:
                n = min(b, args.episodes_per_task - done)
                env_ids = list(range(n))
                states = [init_states[(done + k) % len(init_states)] for k in range(n)]
                frames, succ = _rollout_batch(
                    policy,
                    venv,
                    env_ids,
                    str(task.language),
                    states,
                    resize_size=args.resize_size,
                    replan_steps=args.replan_steps,
                    num_steps_wait=args.num_steps_wait,
                    max_steps=max_steps,
                )
                for k in range(n):
                    for fr in frames[k]:
                        ds.add_frame({**fr, "task": str(task.language)})
                    ds.save_episode()
                    outcomes.append(
                        {
                            "episode_index": ep_index,
                            "suite": args.suite,
                            "task_id": int(task_id),
                            "task": str(task.language),
                            "success": bool(succ[k]),
                            "length": len(frames[k]),
                        }
                    )
                    ep_index += 1
                done += n
                pbar.update(n)
            pbar.close()
        finally:
            venv.close()
        n_ok = sum(o["success"] for o in outcomes if o["task_id"] == task_id)
        n_t = sum(1 for o in outcomes if o["task_id"] == task_id)
        logging.info(f"[{args.suite}#{task_id}] success {n_ok}/{n_t} = {n_ok / max(n_t, 1):.2f}")

    # Sidecar rather than a dataset feature; see the module docstring.
    out = pathlib.Path(ds.root) / "rollout_outcomes.json"
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "suite": args.suite,
        "episodes_per_task": args.episodes_per_task,
        "seed": args.seed,
        "overall_success_rate": float(np.mean([o["success"] for o in outcomes])) if outcomes else None,
        "episodes": outcomes,
    }
    out.write_text(json.dumps(payload, indent=1))
    logging.info(f"wrote {len(outcomes)} episodes -> {ds.root}")
    logging.info(f"overall success {payload['overall_success_rate']}  (failures are the point)")


def _rollout_batch(
    policy, venv, env_ids, task_description, initial_states, *, resize_size, replan_steps, num_steps_wait, max_steps
):
    """Run one batch of episodes, recording every post-wait frame. Structurally mirrors
    `eval_on_libero_vectorized._run_episode_batch` but records observations instead of building a
    replay video, and keeps per-env frame lists so a partial batch still yields whole episodes."""
    import collections

    venv.reset(id=env_ids)
    plans = {e: collections.deque() for e in env_ids}
    obs_stack = venv.set_init_state(init_state=initial_states, id=env_ids)
    obs_by_id = {e: obs_stack[k] for k, e in enumerate(env_ids)}
    frames = {e: [] for e in env_ids}
    success = dict.fromkeys(env_ids, False)
    active = list(env_ids)
    t = 0
    while t < max_steps + num_steps_wait and active:
        if t < num_steps_wait:
            obs_stack, _, _, _ = venv.step([evec.LIBERO_DUMMY_ACTION] * len(active), id=active)
            for k, e in enumerate(active):
                obs_by_id[e] = obs_stack[k]
            t += 1
            continue
        elements = {}
        for e in active:
            el, _ = eval_lib._obs_to_element(obs_by_id[e], task_description, resize_size)  # noqa: SLF001
            elements[e] = el
        need = [e for e in active if not plans[e]]
        if need:
            outs = policy.infer_batch([elements[e] for e in need])
            for e, o in zip(need, outs, strict=True):
                plans[e].extend(o["actions"][:replan_steps])
        acts = {e: np.asarray(plans[e].popleft(), dtype=np.float32) for e in active}
        # Record BEFORE stepping: the frame is the state the action was chosen in.
        for e in active:
            frames[e].append({**_raw_frame(obs_by_id[e]), "actions": acts[e]})
        obs_stack, _, dones, _ = venv.step([acts[e] for e in active], id=active)
        still = []
        for k, e in enumerate(active):
            obs_by_id[e] = obs_stack[k]
            if dones[k]:
                success[e] = True
            else:
                still.append(e)
        active = still
        t += 1
    return [frames[e] for e in env_ids], [success[e] for e in env_ids]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main(tyro.cli(Args))
