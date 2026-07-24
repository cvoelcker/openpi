"""Minimal ENOLCK reproducer. Runs LeRobotDataset(...) once per SLURM task.

Invoke via `sbatch scripts/slurm_train.sh repro_flock.sh` so the environment
(apptainer container, venv, srun layout) matches real training exactly.
"""
import os
import socket
import sys
import time
import traceback

from lerobot.common.datasets import lerobot_dataset

repo_id = os.environ.get("REPRO_REPO_ID", "physical-intelligence/libero")
n_iter = int(os.environ.get("REPRO_ITERS", "3"))
tag = (
    f"[{socket.gethostname()} pid={os.getpid()} "
    f"rank={os.environ.get('SLURM_PROCID', '?')}/{os.environ.get('SLURM_NTASKS', '?')}]"
)

print(
    tag,
    "HF_HOME=", os.environ.get("HF_HOME"),
    "HF_DATASETS_CACHE=", os.environ.get("HF_DATASETS_CACHE"),
    "HF_HUB_CACHE=", os.environ.get("HF_HUB_CACHE"),
    flush=True,
)

for i in range(n_iter):
    t0 = time.time()
    try:
        lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        ds = lerobot_dataset.LeRobotDataset(repo_id)
        print(tag, f"iter={i} ok  len={len(ds)}  dt={time.time() - t0:.2f}s", flush=True)
    except Exception:
        print(tag, f"iter={i} FAIL dt={time.time() - t0:.2f}s", flush=True)
        traceback.print_exc()
        sys.exit(1)
