import os
import argparse

parser = argparse.ArgumentParser(description="Fine-tuning LLaMA with GaLore")
parser.add_argument("--epoch", type=int, default=30, help="Number of training epochs")
args = parser.parse_args()
os.environ["WANDB_PROJECT"]="llm-finetune"

import wandb
from datasets import load_dataset

dataset = load_dataset("gsm8k", "main")

def preprocess_function(examples):
    prompts = ["Question: " + q + "\nAnswer:until " for q in examples['question']]
    responses = examples['answer']
    return {'prompt': prompts, 'response': responses}

tokenized_dataset = dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",    
    torch_dtype = torch.bfloat16,
    attn_implementation = "flash_attention_2",  
    device_map = "auto",
    use_cache = False,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

max_length = 720

def tokenize_function(examples):
    prompt = tokenizer(examples['prompt'])
    response = tokenizer(examples['response'])

    input_ids = prompt.input_ids + response.input_ids + [tokenizer.eos_token_id]
    labels = [-100] * len(prompt.input_ids) + response.input_ids + [tokenizer.eos_token_id]
    attention_mask = prompt.attention_mask + response.attention_mask + [1]
    if len(input_ids) > max_length:
        raise ValueError("The input length is too long for the model.")
    elif len(input_ids) < max_length:
        input_ids += [tokenizer.pad_token_id] * (max_length - len(input_ids))
        labels += [-100] * (max_length - len(labels))
        attention_mask += [0] * (max_length - len(attention_mask))
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
    }

tokenized_dataset = tokenized_dataset.map(tokenize_function, batched=False)

from transformers import Trainer, TrainingArguments

rank = 8
update_proj_gap = 200
scale = 0.25
epoch = args.epoch
training_args = TrainingArguments(
    output_dir = f"logs/{os.environ['SLURM_JOB_ID']}/checkpoint/",
    eval_strategy = "epoch",
    save_strategy = "epoch",
    per_device_train_batch_size = 2,
    #gradient_accumulation_steps=16,
    num_train_epochs = epoch,
    logging_steps = 1,
    learning_rate = 7e-4,
    gradient_checkpointing = True,
    bf16=True,
    lr_scheduler_type="constant",
    optim = "galore_adamw",
    optim_target_modules = "all-linear",
    optim_args = f"rank={rank}, update_proj_gap={update_proj_gap}, scale={scale}",
    report_to="wandb",
    save_total_limit=1,
    run_name=f'galore_ep-{epoch}_rank-{rank}_gap-{update_proj_gap}_sc-{scale}'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
)

trainer.train()

wandb.finish()
