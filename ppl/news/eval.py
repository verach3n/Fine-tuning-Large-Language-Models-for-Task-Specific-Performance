from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from datasets import load_dataset
from tqdm import tqdm
from bert_score import score
import torch
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=int, default=31, help="Checkpoint number")
parser.add_argument("--jobid", type=int, default=3585893, help="Job ID for constructing checkpoint path")
args = parser.parse_args()


batch_size = 1
max_length = 7450
method = "rosa" 

dataset = load_dataset("argilla/news-summary", split="test")
def preprocess_function(examples):
    prompts = [
        f"Article:\n{q}\n\nSummary in English:"
        for q in examples['text']
    ]
    responses = [item[0]['text'] for item in examples['prediction']]
    return {'prompt': prompts, 'response': responses}

dataset = dataset.map(preprocess_function, batched=True)

labels = [item['prediction'][0]['text'] for item in dataset]  # Extracting labels
'''with open("./evals/labels.txt", "w") as label_file:
    for label in labels:
        label_file.write(label + "\n")'''

predictions = []

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  # Adjust the rank value if needed
model = AutoModelForCausalLM.from_pretrained(
    f"./logs/{args.jobid}/checkpoint/checkpoint-{args.ckpt}/",
    #"meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

prompts = [example['prompt'] for example in dataset]
num_batches = len(prompts) // batch_size + (1 if len(prompts) % batch_size != 0 else 0)

print("Generating predictions...") 
for i in tqdm(range(num_batches)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(prompts))
    batch_prompts = prompts[start_idx:end_idx]
    
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            no_repeat_ngram_size=3,
            remove_invalid_values=True,
            pad_token_id=tokenizer.pad_token_id
        )
    batch_summaries = []
    for idx, (output, input_ids) in enumerate(zip(outputs, inputs['input_ids'])):
        input_length = len(input_ids)
        summary = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        print(f"Processing example {start_idx + idx + 1}")
        print(f"Input length: {input_length}")
        print(f"Output length: {len(output)}")
        print(f"Generated summary: {summary}\n")
        batch_summaries.append(summary)

    '''batch_summaries = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]'''

    print(f"\nBatch {i+1}/{num_batches}:")
    for j, summary in enumerate(batch_summaries):
        print(f"\nExample {start_idx + j + 1}:")
        print(f"Prediction: {summary}\n")
        print("-" * 80)
    
    predictions.extend(batch_summaries)

with open(f"./logs/{args.jobid}/predictions.txt", "w", encoding='utf-8') as prediction_file:
    for prediction in predictions:
        prediction_file.write(prediction + "\n")

dataset = dataset.add_column("generated_prediction", predictions)

P, R, F1 = score(predictions, labels, lang="en", verbose=True)

print("BERTScore Precision:", np.mean(P.numpy()))
print("BERTScore Recall:", np.mean(R.numpy()))
print("BERTScore F1:", np.mean(F1.numpy()))

print("\nSaving detailed results...")
with open(f"./logs/{args.jobid}/detailed_results.txt", "w", encoding='utf-8') as f:
    f.write("Detailed Evaluation Results:\n\n")
    for i, (pred, label, p, r, f1) in enumerate(zip(predictions, labels, 
                                                   P.numpy(), R.numpy(), F1.numpy())):
        f.write(f"Example {i+1}:\n")
        f.write(f"Reference: {label}\n")
        f.write(f"Prediction: {pred}\n")
        f.write(f"Precision: {p:.4f}\n")
        f.write(f"Recall: {r:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write("-" * 80 + "\n")

print("Evaluation completed!")