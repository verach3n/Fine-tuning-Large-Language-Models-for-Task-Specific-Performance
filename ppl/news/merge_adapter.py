import os
import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft.tuners.rosa import RosaConfig

def main(
    model_path: str,
    base_model_path: str = "meta-llama/Llama-2-7b-hf",
    device: str = "auto",
    torch_dtype: str = "bfloat16"
):
    print(f"Loading base model from {base_model_path}")
    
    if torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )

    config_path = os.path.join(model_path, "adapter_config.json")
    config = RosaConfig.from_pretrained(model_path)
    
    print(f"Loading ROSA model from {model_path}")
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        config=config, 
        device_map=device
    )

    print("Merging model...")
    model = model.merge_and_unload()

    merged_path = os.path.join(model_path, "merged")
    os.makedirs(merged_path, exist_ok=True)

    print(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(merged_path)

    print("Done!")

if __name__ == "__main__":
    fire.Fire(main)