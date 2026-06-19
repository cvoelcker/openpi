source .venv/bin/activate
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_rep.py pi05_crl_droid_finetune --exp-name=my_experiment --overwrite --data.no-include-next-observation --data.no-include-goal-observation --data.num-parallel-reads=-1 --data.num-parallel-calls=-1 --data.no-filter --data.shuffle-buffer-size=150000 --batch-size=128
