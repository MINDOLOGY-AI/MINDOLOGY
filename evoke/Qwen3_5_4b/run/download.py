# downloads the Qwen3.5-4B checkpoint files the from-scratch text implementation needs into weights/evoke/Qwen3_5_4b
# run from repo root: python -m evoke.Qwen3_5_4b.run.download

import os
from pathlib import Path

# hugging face endpoint ("https://hf-mirror.com" from mainland china); set before huggingface_hub reads it at import
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from huggingface_hub import snapshot_download  # noqa: E402  (must follow the endpoint setting above)

WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "Qwen3_5_4b"
MODEL_ID = "Qwen/Qwen3.5-4B"

# sharded weights are handled via model.safetensors.index.json
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "chat_template.jinja",
]


def main():
    print(f"Downloading to: {WEIGHTS_DIR}")
    snapshot_download(repo_id=MODEL_ID, local_dir=str(WEIGHTS_DIR), allow_patterns=ALLOW_PATTERNS)
    print("Done.")


if __name__ == "__main__":
    main()
