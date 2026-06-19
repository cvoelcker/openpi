#!/bin/bash
#SBATCH -J openpi_train
#SBATCH -o train.%j.out
#SBATCH -e train.%j.err
#SBATCH -N 2                     # Number of nodes
#SBATCH -n 2                     # One task per node
#SBATCH --gpus-per-node=8
#SBATCH -t 48:00:00

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

srun --export=ALL uv run scripts/train.py "$@"
