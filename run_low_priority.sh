#!/bin/bash -l
#SBATCH -D ./
#SBATCH --export=ALL
#SBATCH -J lyapunov_train_low
#SBATCH -o outputs/training_output_%j.out
#SBATCH -e errors/training_error_%j.err
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s-low   # Low priority L40S GPU partition
#SBATCH -N 1
#SBATCH -t 1-00:00:00     # Low priority jobs have a strict maximum limit of 24 hours

module purge
module load miniforge3/25.3.0-python3.12.10

export CONDA_ENVS_PATH=/mnt/fastscratch/users/$USER/.conda/envs
export CONDA_PKGS_DIRS=/mnt/fastscratch/users/$USER/.conda/pkgs
conda activate lyapunov_env

export PYTHONPATH=$(pwd):$(pwd)/libs/sd3/dso

python Lyapunov_test_dso.py
