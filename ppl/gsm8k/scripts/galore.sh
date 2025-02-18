#!/usr/bin/bash
#SBATCH --job-name=galore
#SBATCH -o logs/%A/out.txt
#SBATCH -e logs/%A/err.txt
#SBATCH -t 90:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH -A naiss2024-5-352

source activate env1

EPOCHS=4

python galore.py --epoch $EPOCHS
