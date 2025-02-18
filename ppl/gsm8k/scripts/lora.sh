#!/bin/bash

#SBATCH -A NAISS2024-5-352    # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-03:00:00          # how long time it will take to run
#SBATCH --job-name=lora
#SBATCH --gpus-per-node=A100:1  # choosing no. GPUs and their type
#SBATCH --output=logs/lora/%A/lora.out
#SBATCH --error=logs/lora/%A/lora.err

export RANK=8

# Value 0-1
TRAIN_SPLIT_RATIO=1

source activate galore_simple

python lora.py --train_split_ratio ${TRAIN_SPLIT_RATIO}



