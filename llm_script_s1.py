
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# model_name = "simplescaling/s1.1-32B"
# model_name = "meta-llama/Llama-3.2-1B"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

cache_dir = os.path.join(os.getcwd(), "model_cache")
os.makedirs(cache_dir, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

prompt = "How many r in raspberry"
messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\nGenerated text:\n")
print(response)
print("\n")