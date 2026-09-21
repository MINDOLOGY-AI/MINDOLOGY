"""
Build a local Qwen3.5-4B text model from the generic synapse algorithms and
load the pre-downloaded HF weights.

Weights are expected at:
    MINDOLOGY/weights/evoke/Qwen3_5_4b/

The checkpoint is sharded (model.safetensors.index.json). Text keys
(model.language_model.*) match our module layout exactly; vision-tower keys
and a separate lm_head (embeddings are tied) are dropped, with a printed count.

Tokenizer is loaded via transformers from the same directory.

Run from MINDOLOGY root as a module:
    python -m mind.evoke.Qwen3_5_4b.run.loader
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from synapse.algorithms.transformer.qwen3_5 import Qwen3_5ForCausalLM
from evoke.Qwen3_5_4b.run.config import QWEN3_5_4B_CONFIG


# --- paths ---
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "Qwen3_5_4b"
MODEL_ID = "Qwen/Qwen3.5-4B"


def _resolve_shards(weights_dir):
    """Return the list of .safetensors shard files to load."""
    index_file = weights_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f).get("weight_map", {})
        shards = sorted({weights_dir / name for name in weight_map.values()})
        missing = [s for s in shards if not s.exists()]
        if missing:
            raise FileNotFoundError(f"shard files referenced by {index_file} are missing: {missing}")
        return shards

    shards = sorted(weights_dir.glob("*.safetensors"))
    if shards:
        return shards

    raise FileNotFoundError(f"no safetensors checkpoint found under {weights_dir}")


def load_qwen3_5_model(device=None, dtype=None):
    """
    Build Qwen3.5-4B (text branch) from synapse algorithms and load HF weights.

    device: torch device; default cuda if available else cpu
    dtype:  torch dtype; default bfloat16 on cuda, float32 on cpu
    """
    config = QWEN3_5_4B_CONFIG

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model = Qwen3_5ForCausalLM(config)
    model_keys = set(model.state_dict().keys())

    shards = _resolve_shards(WEIGHTS_DIR)
    print(f"Loading weights from {len(shards)} shard(s) under: {WEIGHTS_DIR}")

    # exact key matching for text keys; everything else (vision tower,
    # tied lm_head) is dropped
    state_dict = {}
    skipped = 0
    for shard in shards:
        print(f"  opening {shard.name}...")
        # Load to CPU first to avoid doubling GPU memory, then copy to model's device.
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in model_keys:
                    tensor = f.get_tensor(key)
                    state_dict[key] = tensor.to(dtype) if tensor.dtype != dtype else tensor
                else:
                    skipped += 1
    print(f"  skipped {skipped} checkpoint keys (vision tower / tied lm_head / other)")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # the tied lm_head intentionally has no checkpoint key; restore the tie
    if config.tie_word_embeddings:
        model.tie_weights()
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing:
        print(f"Warning: missing keys: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected}")

    from evoke.Qwen3_5_4b.run.config import QUANTIZE_INT8
    if QUANTIZE_INT8 and device.type == "cuda":
        from evoke.Qwen3_5_4b.run.quantize import quantize_model_int8
        model = quantize_model_int8(model)

    model.to(device)
    model.eval()
    return model, config


def load_qwen3_5_tokenizer():
    """Load the tokenizer from the pre-downloaded weights directory."""
    tokenizer = AutoTokenizer.from_pretrained(
        str(WEIGHTS_DIR),
        local_files_only=True,
    )
    return tokenizer


def build_model_and_tokenizer(device=None, dtype=None):
    """Convenience wrapper: returns (model, tokenizer, config)."""
    model, config = load_qwen3_5_model(device=device, dtype=dtype)
    tokenizer = load_qwen3_5_tokenizer()
    return model, tokenizer, config


if __name__ == "__main__":
    print("Building model and loading weights...")
    model, tokenizer, config = build_model_and_tokenizer()
    print(f"Model on device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Config: {config}")
