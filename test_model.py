
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "Deepsen8391/qwen25-synthetic-customer-support"

print("Loading tokenizer...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...", flush=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float32,
    low_cpu_mem_usage=True
)

model.eval()

print("Model loaded successfully!", flush=True)

messages = [
    {
        "role": "system",
        "content": "You are a helpful customer support assistant."
    },
    {
        "role": "user",
        "content": "How can I request a refund?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True
)

# Extract input tensors
input_ids = inputs["input_ids"]
attention_mask = inputs.get("attention_mask")

print("Input shape:", input_ids.shape, flush=True)

with torch.inference_mode():
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

new_tokens = output[0, input_ids.shape[-1]:]

answer = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
)

print("\nGenerated answer:")
print(answer)
