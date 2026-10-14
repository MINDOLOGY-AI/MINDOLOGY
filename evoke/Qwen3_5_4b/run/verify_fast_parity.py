# fast kernels (load_qwen3_5_model(fast=True): fla's gated deltanet kernels + sdpa) vs the torch reference path on one
# bf16 model: the same real chunks through both, logits compared, forward time per batch measured, and greedy decode
# (fla's fused recurrent kernel) compared.
# needs a gpu (~12GB) and fla-core. run from repo root:
#   python -m evoke.Qwen3_5_4b.run.verify_fast_parity

import json
import time
from pathlib import Path

import numpy as np
import torch

from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_model, load_qwen3_5_tokenizer

BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "qwen3_5_4b_interp_dataset"
# the sae trainer's step: 64 chunks x 128 tokens
BATCH_CHUNKS = 64
N_TIMED = 5
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 16


def set_fast(model, fast):
    # flips both attention kinds of every layer between the fast kernels and the torch reference
    for layer in model.model.language_model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.fast = fast
        else:
            layer.self_attn.use_flash = fast


def main():
    model = load_qwen3_5_model(device=torch.device("cuda"), fast=True)[0]
    meta = json.loads((BIN_DIR / "meta.json").read_text())
    # (BATCH_CHUNKS, chunk_size) the first eval chunks
    ids = torch.from_numpy(np.fromfile(BIN_DIR / "eval.bin", dtype=np.int32, count=BATCH_CHUNKS * meta["chunk_size"])
                           .reshape(BATCH_CHUNKS, -1).astype(np.int64)).cuda()
    logits = {}  # {fast: (BATCH_CHUNKS, chunk_size, V) fp32}
    for fast in (False, True):
        set_fast(model, fast)
        with torch.no_grad():
            logits[fast] = model(input_ids=ids, use_cache=False)[0].float()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(N_TIMED):
                model(input_ids=ids, use_cache=False)
            torch.cuda.synchronize()
        print(f"fast={fast}: {(time.time() - t) / N_TIMED * 1000:.0f} ms per {BATCH_CHUNKS}x{meta['chunk_size']} forward", flush=True)
    diff = (logits[True] - logits[False]).abs()
    agree = (logits[True].argmax(-1) == logits[False].argmax(-1)).float().mean().item()
    # (B, T, V) -> mean KL(slow || fast) per token
    kl = torch.nn.functional.kl_div(logits[True].log_softmax(-1), logits[False].log_softmax(-1), log_target=True, reduction="none").sum(-1).mean().item()
    print(f"logits |diff| max {diff.max().item():.3f} mean {diff.mean().item():.4f}, top-1 agreement {agree:.4f}, KL {kl:.2e}")

    tok = load_qwen3_5_tokenizer()
    outs = {}  # {fast: str} greedy continuation with the hybrid cache
    for fast in (False, True):
        set_fast(model, fast)
        x = torch.tensor([tok.encode(PROMPT)], device="cuda")
        past = None
        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                step = x if past is None else x[:, -1:]
                out, _, past = model(input_ids=step, past_key_values=past, use_cache=True)
                x = torch.cat([x, out[:, -1].argmax(-1, keepdim=True)], dim=-1)
        outs[fast] = tok.decode(x[0].tolist())
        print(f"fast={fast} greedy: {outs[fast]!r}")
    assert agree > 0.99, f"top-1 agreement {agree:.4f}: the fast path does not match"


if __name__ == "__main__":
    main()
