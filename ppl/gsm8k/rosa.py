from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, TaskType
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, TaskType
from peft.tuners.rosa import RosaConfig, RosaScheduler
import torch

os.environ["WANDB_PROJECT"]="llm-finetune"

dataset = load_dataset("gsm8k", "main", split="train")

def preprocess_function(examples):
    prompts = ["Question: " + q + "\nAnswer:until " for q in examples['question']]
    responses = examples['answer']
    return {'prompt': prompts, 'response': responses}

'''dataset = load_dataset("argilla/news-summary", split="train")

def preprocess_function(examples):
    #prompts = examples['text']
    prompts = [
        f"Article:\n{q}\n\nSummary in English:"
        for q in examples['text']
    ]
    responses = [item[0]['text'] for item in examples['prediction']]
    return {'prompt': prompts, 'response': responses}'''

tokenized_dataset = dataset.map(preprocess_function, batched=True)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype = torch.bfloat16,
    #attn_implementation = "flash_attention_2",
    device_map = "auto",
    use_cache = False,
)
device = torch.device("cuda:0")
memory_stats = torch.cuda.memory_stats(device)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

max_length = 720

rank = 8
peft_config = RosaConfig(
    r=rank,
    lora_alpha=16,
    target_modules="all-linear",
    bias="none",
    task_type=TaskType.CAUSAL_LM,

    d=0.006,
    spa_num_grads=1,
    grad_acc_mode='mean_squared',
    schedule='wl4'        
)

model = get_peft_model(model, peft_config)

#model.print_trainable_parameters()

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

training_args = TrainingArguments(
    output_dir=f"./llama2-gsm8k-rosa/r{rank}/",
    per_device_train_batch_size=1,
  #  per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    evaluation_strategy="no",
    bf16=False,
    fp16=False,
    save_strategy="epoch",
 #   load_best_model_at_end=True,
    learning_rate=7e-4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
#    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
    callbacks=[RosaScheduler(model)]
)

trainer.train()


wandb.finish()
