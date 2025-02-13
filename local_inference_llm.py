#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script performs local inference using the small-scale language model via the Hugging Face API (e.g., meta-llama/Llama-3.2-1B or deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).
It now supports device selection: auto, cuda, mps (Apple Silicon), or cpu.

Usage:
    python3 local_inference_llama.py --model <model-name> --prompt "Your prompt here" [--max_length 100] [--device auto]

Example:
    python3 local_inference_llama.py --model meta-llama/Llama-3.2-1B --prompt "Introduce Taiwanese cuisine" --max_length 1500 --device auto
    python3 local_inference_llama.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --prompt "Introduce Taiwanese cuisine" --max_length 1500 --device auto
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device(device_option: str) -> torch.device:
    """
    Determines which device to use based on the device_option argument.
    Options:
        - "auto": Automatically choose CUDA if available, then MPS if available, else CPU.
        - "cuda": Use CUDA if available, otherwise fall back to CPU.
        - "mps": Use Apple's MPS if available, otherwise fall back to CPU.
        - "cpu": Force CPU.
    """
    device_option = device_option.lower()
    if device_option == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    elif device_option == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_option == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print("MPS is not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_option == "cpu":
        return torch.device("cpu")
    else:
        print(f"Invalid device option '{device_option}'. Falling back to CPU.")
        return torch.device("cpu")

def load_model(model_name: str, device: torch.device):
    """
    Loads the specified model and tokenizer from the Hugging Face Hub.
    If using a non-CPU device, the model is loaded with torch.float16 to save memory.
    """
    print(f"Loading model: {model_name} on device: {device}")

    cache_dir = os.path.join(os.getcwd(), "model_cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Download and load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    # Set data type: use float32 for CPU; for CUDA/MPS, float16 may be used
    torch_dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        cache_dir=cache_dir
    )
    model.to(device)
    return tokenizer, model

def generate_text(tokenizer, model, prompt: str, max_length: int, device: torch.device) -> str:
    """
    Generates text based on the input prompt.
    """
    # Tokenize the prompt and move tensors to the specified device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate text without computing gradients (inference mode)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    
    # Decode the generated tokens to a string
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Local inference for language model with device selection")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        help="Hugging Face model name or path")
    parser.add_argument("--prompt", type=str, default="This is a simple local LLM inference test. Wait!",
                        help="Prompt text for generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length for the generated text")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to use for inference: auto, cuda, mps, or cpu")
    args = parser.parse_args()

    # Determine the device to use based on the user's option
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer, model = load_model(args.model, device)
    print("\nGenerating text...")
    result = generate_text(tokenizer, model, args.prompt, max_length=args.max_length, device=device)
    print("\nGenerated text:\n")
    print(result)
    print("\n")

if __name__ == '__main__':
    main()
