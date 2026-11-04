# shared pieces of the LSCL baseline study (design: original_research/lscl/LSCL.md, pipeline: README.md here):
# paths, model loading, qwen's chat prompt, batched greedy generation and the fact match rule.

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_model, load_qwen3_5_tokenizer

SEED = 21
# model-agnostic pools: princeton facts and the general benchmarks / prompt sources (datasteps/LSCL/*/doc.md)
POPQA = Path("data/datasteps/LSCL/princeton/popqa.jsonl")
GENERAL = Path("data/datasteps/LSCL/general")
# this model's data: pretest, splits, GENERAL_DATASET, augmented facts (schemas in README.md)
DATA = Path("data/lscl/qwen3_5_4b")
# per-run results (metrics + per-fact predictions) and the base model's
RESULTS = Path("results/Qwen3_5_4b/lscl")
# per-run final weights and the EWC fisher
WEIGHTS = Path("weights/lscl/qwen3_5_4b")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load(dtype=torch.bfloat16):
    # -> qwen3.5-4b on cuda (fla + sdpa kernels) in dtype, tokenizer. bf16 for inference; fp32 master weights for training
    model, _ = load_qwen3_5_model(device=torch.device("cuda"), dtype=torch.bfloat16, fast=True)
    model = model.to(dtype)
    model.eval()
    tok = load_qwen3_5_tokenizer()
    assert tok.eos_token == "<|im_end|>" and tok.pad_token_id is not None
    return model, tok


def chat_prompt(tok, user):
    # one user turn in qwen's chat template, thinking off (the template writes an empty think block), ready for the reply
    return tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, enable_thinking=False, tokenize=False)


@torch.no_grad()
def generate(model, tok, prompts, max_new, batch_tokens=131072, max_rows=256):
    # prompts: [str] already templated -> [(reply str, finished bool)] greedy, in prompt order. finished = the reply
    # ended with <|im_end|> (False: cut at max_new). prompts are sorted by length and packed into left-padded batches of
    # at most batch_tokens (prompt + max_new) tokens and max_rows rows (each row's gated deltanet state is ~50 MB fp32:
    # 24 layers x 32 heads x 128 x 128); the hybrid cache handles the left padding.
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
    stop = torch.tensor([tok.eos_token_id, tok.pad_token_id], device="cuda")
    out = [None] * len(ids)  # [(str, bool)]
    i = 0
    while i < len(order):
        # the first (longest) prompt sets the padded length of the batch
        n = max(1, min(max_rows, batch_tokens // (len(ids[order[i]]) + max_new)))
        rows = order[i:i + n]
        i += len(rows)
        T = len(ids[rows[0]])
        # (B, T) left padded ids and their mask
        cur = torch.tensor([[tok.pad_token_id] * (T - len(ids[r])) + ids[r] for r in rows], device="cuda")
        mask = torch.tensor([[0] * (T - len(ids[r])) + [1] * len(ids[r]) for r in rows], device="cuda")
        past = None
        done = torch.zeros(len(rows), dtype=torch.bool, device="cuda")
        new = []  # [(B,) token ids] one per step
        for _ in range(max_new):
            # (B, 1, vocab)
            logits, _, past = model(input_ids=cur, attention_mask=mask, past_key_values=past, use_cache=True, logits_to_keep=1)
            # (B,)
            nxt = torch.where(done, tok.pad_token_id, logits[:, -1].argmax(-1))
            new.append(nxt)
            done |= torch.isin(nxt, stop)
            if done.all():
                break
            cur = nxt[:, None]
            mask = torch.cat([mask, torch.ones(len(rows), 1, dtype=mask.dtype, device="cuda")], 1)
        # (B, n_new)
        for r, toks in zip(rows, torch.stack(new, 1).tolist()):
            cut = next((k for k, t in enumerate(toks) if t in (tok.eos_token_id, tok.pad_token_id)), None)
            out[r] = (tok.decode(toks[:cut], skip_special_tokens=True).strip(), cut is not None)
    return out


def normalize(text):
    # lowercase, every run of non-alphanumerics -> one space, padded with spaces so " alias " matches whole words only
    return " " + re.sub(r"[^0-9a-z]+", " ", text.lower()).strip() + " "


def fact_match(reply, aliases):
    # the fact match rule of pretest and every eval: some accepted alias appears in the reply as whole words
    # ("pol" matches "a pol" but not "napoleon"). aliases that normalize to nothing (pure punctuation) never match
    reply = normalize(reply)
    return any(normalize(a) != "  " and normalize(a) in reply for a in aliases)


def run_program(program, timeout_s=10):
    # runs python source in a fresh process inside a scratch directory -> True if it exits 0 within timeout_s
    with tempfile.TemporaryDirectory() as tmp:
        try:
            return subprocess.run([sys.executable, "-c", program], cwd=tmp, capture_output=True, timeout=timeout_s).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def code_block(reply):
    # the reply's first ``` code block, else the whole reply
    m = re.search(r"```(?:python)?\n(.*?)```", reply, flags=re.DOTALL)
    return m.group(1) if m else reply
