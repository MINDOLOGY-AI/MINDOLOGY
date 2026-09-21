import os
from pathlib import Path

# --- CHINA MIRROR ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- WHERE TO STORE WEIGHTS ---
# Option A: relative to CWD (where you run `python script.py` from)
WEIGHTS_DIR = Path.cwd() / "weights/evoke/Qwen3_5_4b"


# --- DOWNLOAD ---
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3.5-4B"

# only the files the from-scratch text implementation needs;
# sharded weights are handled via model.safetensors.index.json
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "chat_template.jinja",
]

print(f"Downloading to: {WEIGHTS_DIR.resolve()}")
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(WEIGHTS_DIR),
    allow_patterns=ALLOW_PATTERNS,
)

print("Done.")
