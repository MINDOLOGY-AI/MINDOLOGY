import os
from pathlib import Path

# --- CHINA MIRROR ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- WHERE TO STORE WEIGHTS ---
# Option A: relative to CWD (where you run `python script.py` from)
WEIGHTS_DIR = Path.cwd() / "weights/evoke/OlMo2_1b"


# --- DOWNLOAD ---
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"

print(f"Downloading to: {WEIGHTS_DIR.resolve()}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(WEIGHTS_DIR))
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=str(WEIGHTS_DIR))

print("Done.")
