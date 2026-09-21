"""
Build a local OlMo2-1B model from the generic synapse algorithms and load the
pre-downloaded HF weights.

Weights are expected at:
    MINDOLOGY/weights/evoke/OlMo2_1b/

Tokenizer is loaded via transformers from the same cache directory.
"""

import importlib.util
import sys
from pathlib import Path

# make synapse importable when running from project root
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from synapse.algorithms.transformer.olmo2 import Olmo2ForCausalLM

# Load config from this same hyphenated folder via importlib because Python
# packages cannot contain '-'.
CONFIG_PATH = PROJECT_ROOT / "evoke" / "OlMo2_1b" / "run" / "config.py"
spec = importlib.util.spec_from_file_location("olmo2_1b_config", CONFIG_PATH)
olmo2_1b_config = importlib.util.module_from_spec(spec)
sys.modules["olmo2_1b_config"] = olmo2_1b_config
spec.loader.exec_module(olmo2_1b_config)
OLMO2_1B_INSTRUCT_CONFIG = olmo2_1b_config.OLMO2_1B_INSTRUCT_CONFIG


# --- paths ---
WEIGHTS_DIR = PROJECT_ROOT / "weights" / "evoke" / "OlMo2_1b"
MODEL_NAME = "allenai/OLMo-2-0425-1B-Instruct"


def load_olmo2_model(device=None, dtype=None, strict=True):
    """
    Build OlMo2-1B-Instruct from synapse algorithms and load HF weights.

    device: torch device; default cuda if available else cpu
    dtype:  torch dtype; default bfloat16 on cuda, float32 on cpu
    strict: passed to load_state_dict
    """
    config = OLMO2_1B_INSTRUCT_CONFIG

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = Olmo2ForCausalLM(config).to(device=device, dtype=dtype)

    # find the cached safetensors file
    safetensors_path = WEIGHTS_DIR / "models--allenai--OLMo-2-0425-1B-Instruct" / "snapshots"
    safetensors_files = list(safetensors_path.rglob("model.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"Could not find model.safetensors under {safetensors_path}")
    safetensors_file = safetensors_files[0]
    print(f"Loading weights from: {safetensors_file}")

    # Load to CPU first to avoid doubling GPU memory, then copy to model's device.
    hf_state = load_file(safetensors_file, device="cpu")

    # dtype conversion
    mapped = {k: (v.to(dtype) if v.dtype != dtype else v) for k, v in hf_state.items()}
    del hf_state

    missing, unexpected = model.load_state_dict(mapped, strict=strict)
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")

    model.eval()
    return model, config


def load_olmo2_tokenizer():
    """Load the tokenizer from the pre-downloaded HF cache."""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=str(WEIGHTS_DIR),
        local_files_only=True,
    )
    return tokenizer


def build_model_and_tokenizer(device=None, dtype=None):
    """Convenience wrapper: returns (model, tokenizer, config)."""
    model, config = load_olmo2_model(device=device, dtype=dtype)
    tokenizer = load_olmo2_tokenizer()
    return model, tokenizer, config


if __name__ == "__main__":
    print("Building model and loading weights...")
    model, tokenizer, config = build_model_and_tokenizer()
    print(f"Model on device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Config: {config}")
