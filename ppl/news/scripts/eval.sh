#!/bin/bash

#SBATCH -A NAISS2024-5-352    # find your project with the "projinfo" command
#SBATCH -p alvis               # what partition to use (usually not necessary)
#SBATCH -t 0-10:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A100fat:1  # choosing no. GPUs and their type
#SBATCH --output=logs/3585893/eval.out
#SBATCH --error=logs/3585893/eval.err

CKPT=31
JOBID=3585893
export BASE="logs/${JOBID}/checkpoint/checkpoint-${CKPT}"

source activate rosa

python merge_adapter.py ${BASE}

python eval.py \
    --ckpt ${CKPT} \
    --jobid ${JOBID}
