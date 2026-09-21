"""
Verify the local Qwen3.5-4B reimplementation against the HF reference.

Loads both models from the same local checkpoint, runs the same prompt
through both, and compares logits + greedy generations. If the port is
correct, logits match to bf16 tolerance.

QUANTIZE_INT8 must be False — int8 noise would drown the comparison.

Run from MINDOLOGY root as a module (needs ~16GB VRAM for two bf16 4Bs,
run on a rented box if the laptop can't fit both):
    python -m mind.evoke.Qwen3_5_4b.run.verify_hf_parity
"""

import torch
from transformers import AutoModelForCausalLM

from evoke.Qwen3_5_4b.run.config import QUANTIZE_INT8
assert not QUANTIZE_INT8, "set QUANTIZE_INT8 = False in config.py before verifying parity"

from evoke.Qwen3_5_4b.run.loader import (
    WEIGHTS_DIR,
    load_qwen3_5_model,
    load_qwen3_5_tokenizer,
)

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
# bf16 accumulation noise shows up around 1e-2 on raw logits; anything past
# this means a real architecture mismatch, not rounding
MAX_TOLERATED_LOGIT_DIFF = 0.1


def greedy_generate_local(model, input_ids, max_new):
    # greedy decode on the local model using its hybrid cache
    past = None
    ids = input_ids
    for _ in range(max_new):
        with torch.no_grad():
            step_ids = ids if past is None else ids[:, -1:]
            logits, _, past = model(input_ids=step_ids, past_key_values=past, use_cache=True)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch, 1)
        ids = torch.cat([ids, next_id], dim=-1)
    return ids


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = load_qwen3_5_tokenizer()
    local_model, _ = load_qwen3_5_model(device=device, dtype=dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(WEIGHTS_DIR), local_files_only=True, torch_dtype=dtype
    ).to(device).eval()

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)  # (1, seq)

    # --- logit comparison on the shared prompt ---
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits  # (1, seq, vocab)
        local_logits, _, _ = local_model(input_ids=input_ids, use_cache=False)

    diff = (hf_logits - local_logits).abs()
    max_diff = diff.max().item()
    hf_top1 = hf_logits[:, -1, :].argmax().item()
    local_top1 = local_logits[:, -1, :].argmax().item()

    print(f"\n--- logit parity on prompt: {PROMPT!r} ---")
    print(f"max abs diff:   {max_diff:.6f}  (tolerance {MAX_TOLERATED_LOGIT_DIFF})")
    print(f"mean abs diff:  {diff.mean().item():.6f}")
    print(f"next-token top1: hf={hf_top1} local={local_top1} match={hf_top1 == local_top1}")

    # --- greedy generation comparison ---
    with torch.no_grad():
        hf_ids = hf_model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    local_ids = greedy_generate_local(local_model, input_ids, MAX_NEW_TOKENS)

    print(f"\n--- greedy generation ({MAX_NEW_TOKENS} tokens) ---")
    print(f"hf:    {tokenizer.decode(hf_ids[0, input_ids.shape[1]:])!r}")
    print(f"local: {tokenizer.decode(local_ids[0, input_ids.shape[1]:])!r}")

    assert max_diff < MAX_TOLERATED_LOGIT_DIFF, f"PARITY FAIL: max logit diff {max_diff}"
    assert hf_top1 == local_top1, "PARITY FAIL: next-token disagreement"
    print("\nPARITY OK")


if __name__ == "__main__":
    main()
