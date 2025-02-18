from datasets import load_dataset
import os
import argparse
import wandb

os.environ["WANDB_PROJECT"]="llm-finetune"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_split_ratio",
    type=float,
    default=0.5,
)
args = parser.parse_args()


dataset = load_dataset("gsm8k", "main")

seed = 42
original_train_len = len(dataset["train"])
new_train_len = int(original_train_len * args.train_split_ratio)
dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(new_train_len))

def preprocess_function(examples):
    prompts = ["Question: " + q + "\nAnswer:until " for q in examples['question']]
    responses = examples['answer']
    return {'prompt': prompts, 'response': responses}

tokenized_dataset = dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
'''model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype = torch.bfloat16,
    attn_implementation = "flash_attention_2",
    device_map = "auto",
    use_cache = False,
)'''
compute_dtype = getattr(torch, "bfloat16")              # Use bfloat16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)
base_model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    torch_dtype = torch.bfloat16,
    device_map={"": 0}
)
device = torch.device("cuda:0")
memory_stats = torch.cuda.memory_stats(device)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

rank = int(os.environ.get('RANK'))

lora_config = LoraConfig(
    r=rank,
    lora_alpha=16,
    target_modules="all-linear",
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

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

training_args = TrainingArguments(
    output_dir=f"./logs/lora/{os.environ['SLURM_JOB_ID']}/checkpoint/",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    evaluation_strategy="epoch",
    fp16=True,
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=7e-4,
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
