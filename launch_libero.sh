source .venv/bin/activate

args=(
  pi05_crl_libero_finetune
  --exp-name=crl_rep_train              # checkpoint/W&B run name
  --overwrite                           # replace existing checkpoint dir
  # --data.no-include-next-observation    # skip unused HER next obs
  # --data.no-include-goal-observation    # skip unused HER goal obs
  # --data.num-parallel-reads=8           # 8 is the verified max on 1 node so far
  # --data.num-parallel-calls=8           # 8 is the verified max on 1 node so far
  # --data.no-filter                      # disable DROID filter ranges
  # --data.shuffle-buffer-size=50000      # 50000 won't OOM on 1 node
  --batch-size=64
  # --save-interval=5000                # skip periodic save (will save at last train step)
  # --num-train-steps=25000
)
# python scripts/compute_norm_stats.py --config-name="${args[0]}"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_rep.py "${args[@]}"
