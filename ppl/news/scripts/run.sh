#!/bin/bash

#SBATCH -A NAISS2024-5-352    # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-03:00:00          # how long time it will take to run
#SBATCH --job-name=news
#SBATCH --gpus-per-node=A100fat:1  # choosing no. GPUs and their type
#SBATCH --output=logs/%A/out.out
#SBATCH --error=logs/%A/err.err

source activate env1

python lora.py 
