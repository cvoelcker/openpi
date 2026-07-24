source .venv/bin/activate

# Optional: redirect HF cache to node-local storage to test whether the shared
# scratch filesystem is the flock culprit. Enable with REPRO_LOCAL_CACHE=1.
if [ "${REPRO_LOCAL_CACHE:-0}" = "1" ]; then
  export HF_HOME="${SLURM_TMPDIR:-/tmp/$USER}/hf"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HF_HUB_CACHE="$HF_HOME/hub"
  mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
fi

# Quick flock probe on the filesystems that matter, from inside the container.
python - <<'PY'
import fcntl, os, tempfile
for base in ("/scratch", "/tmp", os.environ.get("SLURM_TMPDIR", "/tmp"), os.path.expanduser("~")):
    try:
        d = tempfile.mkdtemp(prefix="flockprobe_", dir=base)
        with open(os.path.join(d, "x"), "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            fcntl.flock(f, fcntl.LOCK_UN)
        print(f"flock on {base}: OK")
    except OSError as e:
        print(f"flock on {base}: FAIL {e}")
PY

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/repro_flock.py
