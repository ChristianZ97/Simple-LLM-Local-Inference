#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script performs local inference using the small-scale language model via the Hugging Face API.

Supports:
- Device selection: auto, cuda, mps, cpu
- Quantization: int4, int8, fp16, fp32 (default: fp16)
- Efficient memory allocation with `device_map="auto"`

Usage:
    python3 local_inference_llm.py --model <model-name> --prompt "Your prompt here" [--max_length 100] [--quantization fp16]
Example:
    python3 local_inference_llm.py --model meta-llama/Llama-3.2-1B --prompt "How many r's in Strawberry?" --max_length 5000 --quantization int4
    python3 local_inference_llm.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --prompt "Let's think step-by-step. How many r's in Strrawbery? There are" --max_length 1000 --quantization fp16

    python3 local_inference_llm.py --model meta-llama/Llama-3.2-3B --prompt "Think step-by-step. How many r's in Strrawberrier?" --max_length 5000 --quantization fp32
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def get_device() -> torch.device:
    """Selects the best available device based on user preference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_model(model_name: str, device: torch.device, quantization: str):
    """Loads the model and tokenizer with selected quantization."""
    print(f"Loading model: {model_name} on device: {device} with quantization: {quantization}")

    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Load tokenizer (some models require trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)

    # Select quantization type
    quant_config = None
    torch_dtype = torch.float32  # Default to FP32

    if quantization == "int4":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quantization == "int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "fp16":
        torch_dtype = torch.float16 if device.type != "cpu" else torch.float32
    elif quantization == "fp32":
        torch_dtype = torch.float32
    else:
        print(f"Warning: Unknown quantization type '{quantization}', falling back to fp16.")
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        device_map="auto",  # Automatically assign device
        cache_dir=cache_dir
    )

    return tokenizer, model

def generate_text(tokenizer, model, prompt: str, max_length: int, device: torch.device) -> str:
    """Generates text based on the input prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_length,  # Avoid exceeding model's max length
            pad_token_id=tokenizer.eos_token_id  # Avoid warnings
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def main():
    parser = argparse.ArgumentParser(description="Optimized Local LLM Inference with Quantization Options")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Hugging Face model name or path")
    parser.add_argument("--prompt", type=str, default="This is a simple local LLM inference test. Wait!",
                        help="Prompt text for generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum tokens for the generated text")
    parser.add_argument("--quantization", type=str, default="fp16", choices=["int4", "int8", "fp16", "fp32"],
                        help="Quantization method: int4, int8, fp16 (default), or fp32")

    args = parser.parse_args()
    device = get_device()

    print(f"Using device: {device}")
    tokenizer, model = load_model(args.model, device, quantization=args.quantization)

    print("\nGenerating text...")
    result = generate_text(tokenizer, model, args.prompt, max_length=args.max_length, device=device)
    
    print("\nGenerated text:\n")
    print(result)
    print("\n")

if __name__ == '__main__':
    main()
