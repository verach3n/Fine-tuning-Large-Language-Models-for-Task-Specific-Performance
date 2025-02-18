#!/usr/bin/bash
#SBATCH --job-name=rosa
#SBATCH --output=logs/%A/rosa.out
#SBATCH --error=logs/%A/rosa.err
#SBATCH -t 16:00:00
#SBATCH --gpus-per-node=A100fat:1
#SBATCH --nodes=1
#SBATCH -A naiss2024-5-352

source activate ppl

python rosa.py