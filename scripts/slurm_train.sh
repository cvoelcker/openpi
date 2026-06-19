#!/bin/bash
#SBATCH -J openpi_train
#SBATCH -o train.%j.out
#SBATCH -e train.%j.err
#SBATCH -N 4                     # Number of nodes
#SBATCH -n 4                     # One task per node
#SBATCH -t 4:00:00
#SBATCH -p gh
#SBATCH -A ASC26008

# Usage:
#   sbatch scripts/slurm_train.sh <config_name> --exp-name=<name> [extra flags...]
#
# Example:
#   sbatch --nodes=4 scripts/slurm_train.sh pi05_libero --exp-name=my_run --fsdp-devices=8
#
# JAX multi-node discovery uses these SLURM env vars automatically:
#   SLURM_JOB_ID, SLURM_STEP_NODELIST, SLURM_NODELIST, etc.
#
# Make sure XLA_PYTHON_CLIENT_MEM_FRACTION is set for GPU memory allocation.

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# srun --export=ALL uv run scripts/train.py "$@"
# srun --export=ALL uv run scripts/train_rep.py pi05_100_droid_full_finetune_8 --exp-name=my_experiment --overwrite --data.no-filter --fsdp_devices=1 --batch_size=16

srun --export=ALL bash -c '
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
  module load tacc-apptainer
  apptainer exec --nv tensorflow_2502.sif bash ./launch.sh
  '
