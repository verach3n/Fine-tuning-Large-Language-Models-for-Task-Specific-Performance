#!/usr/bin/bash
#SBATCH --job-name=eval_gsm
#SBATCH -o logs/3584997/eval.out.txt
#SBATCH -e logs/3584997/eval.err.txt
#SBATCH -t 1:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH -A naiss2024-5-352

JOBID=3584997
CKPT=233
# Modify the path
export BASE="logs/${JOBID}/checkpoint/checkpoint-${CKPT}"

source activate env2

# Comment this line if you are running with GaLore
python merge_adapter.py ${BASE}

# Modify the values
python eval.py \
    --ckpt ${CKPT} \
    --jobid ${JOBID} \
    --output logs/lora/${JOBID}/gsm8k_eval_output.json

