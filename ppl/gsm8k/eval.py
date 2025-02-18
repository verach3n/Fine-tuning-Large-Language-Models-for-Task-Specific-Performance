import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def extract_final_answer(text):
    
    marker = "####"
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()
    else:
        return ""

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K test set on a given checkpoint using batched generation and compute accuracy. "
                    "For both prediction and ground truth, only the part after '####' is considered as the answer."
    )
    parser.add_argument("--ckpt", type=int, default=56055, help="Checkpoint number")
    parser.add_argument("--jobid", type=int, default=3564101, help="Job ID for constructing checkpoint path")
    parser.add_argument("--output", type=str, default="gsm8k_eval_output.json", help="Output JSON file path")
    parser.add_argument("--max_length", type=int, default=500, help="Max generation length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation")
    args = parser.parse_args()

    checkpoint_path = f"logs/{args.jobid}/checkpoint/checkpoint-{args.ckpt}"
    print(f"Loading checkpoint from: {checkpoint_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True
    )
    model.eval()

    dataset = load_dataset("gsm8k", "main")
    test_dataset = dataset["test"]

    def preprocess(sample):
        sample["prompt"] = "Question: " + sample["question"].strip() + "\nAnswer:until "
        return sample

    test_dataset = test_dataset.map(preprocess)
    test_samples = list(test_dataset)
    total = len(test_samples)
    print(f"Total test samples: {total}", flush=True)

    results_list = []
    correct_count = 0
    bs = args.batch_size

    for i in tqdm(range(0, total, bs), desc="Evaluating"):
        batch_samples = test_samples[i: i+bs]
        prompts = [sample["prompt"] for sample in batch_samples]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=args.max_length,
                do_sample=False 
            )

   
        generated_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        for j, sample in enumerate(batch_samples):
            prompt_text = sample["prompt"]
            full_generated = generated_texts[j]
            
            generated_content = full_generated[len(prompt_text):].strip()
            
            pred_answer = extract_final_answer(generated_content)
            expected_answer = extract_final_answer(sample["answer"])
            acc = (pred_answer == expected_answer)
            if acc:
                correct_count += 1

            result = {
                "doc_id": sample.get("problem_id", i+j),
                "prompt": prompt_text,
                "generated_full": full_generated, 
                "predicted_final_answer": pred_answer,
                "expected_final_answer": expected_answer,
                "acc": acc
            }
            results_list.append(result)
            
            print(f"Sample {i+j+1}: Generated: '{full_generated}' \n | Predicted: '{pred_answer}' \n | Expected: '{expected_answer}' \n | Correct: {acc} \n\n ", flush=True)

    overall_accuracy = correct_count / total if total > 0 else 0.0
    output_data = {
        "accuracy": overall_accuracy,
        "correct": correct_count,
        "total": total,
        "results": results_list
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Evaluation completed. Accuracy: {overall_accuracy:.4f} ({correct_count}/{total})", flush=True)
    print(f"Results saved to {args.output}", flush=True)

if __name__ == "__main__":
    main()
