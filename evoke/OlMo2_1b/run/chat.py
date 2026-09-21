from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- CONFIG ---
WEIGHTS_DIR = Path.cwd() / "weights/evoke/OlMo2_1b"
MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"

# --- LOAD ---
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=str(WEIGHTS_DIR),
    local_files_only=True,
)

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir=str(WEIGHTS_DIR),
    local_files_only=True,
    dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

print(f"Model loaded on: {device}\n")

messages = []
print("OLMo-1B chat. Type 'quit' to exit.\n")

while True:
    user_text = input("You: ").strip()
    if user_text.lower() in {"quit", "exit", "q"}:
        break

    messages.append({"role": "user", "content": user_text})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    print("Generating...", end=" ", flush=True)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,              # <-- pass the tensor, not BatchEncoding
            attention_mask=inputs.attention_mask,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("done")

    reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Bot: {reply}\n")
    messages.append({"role": "assistant", "content": reply})
