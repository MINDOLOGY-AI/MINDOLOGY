# shared pieces of the LSCL pipeline on olmo2-1b-instruct (see original_research/lscl/princeton_benchmark.md, "how we do it here"):
# the one prompt template used at filter / train / eval time, answer normalization for the filter, and batched greedy decoding.

import json
import re
import string
from pathlib import Path

import torch

POOL_DIR = Path.cwd() / "data" / "datasteps" / "LSCL" / "princeton"
OUT_DIR = Path.cwd() / "data" / "lscl" / "olmo2_1b"

# olmo2-instruct chat template (tokenizer_config.json), bos is the literal <|endoftext|> string
PROMPT = "<|endoftext|><|user|>\nQuestion: {question}\nThe answer is:\n<|assistant|>\n"

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = str.maketrans("", "", string.punctuation)


def load_facts(path):
    # -> [{"id", "question", "answer", "aliases", "subject", "relation", "meta"}]
    return [json.loads(l) for l in open(path)]


def normalize(text):
    # squad-style: first line only, lowercase, no punctuation, no articles, single spaces
    text = text.strip().split("\n")[0].lower().translate(PUNCT)
    return " ".join(ARTICLES.sub(" ", text).split())


@torch.inference_mode()
def generate_greedy(model, tokenizer, prompts, max_new_tokens, batch_size):
    # prompts: [str] -> [str] greedy continuation per prompt, cut at eos, in the same order.
    # prompts are batched by identical token length so no padding is ever needed: the synapse causal mask has no
    # fix for fully padded query rows (they softmax to nan), and same-length batches make the problem not exist.
    device = next(model.parameters()).device
    eos = tokenizer.eos_token_id
    # [[int]] per prompt, template already contains the bos string so no special tokens added
    encoded = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    by_len = {}  # {token length: [prompt index]}
    for i, ids in enumerate(encoded):
        by_len.setdefault(len(ids), []).append(i)

    outputs = [None] * len(prompts)  # [str]
    done = 0
    for length in sorted(by_len):
        idxs = by_len[length]
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start:start + batch_size]
            # (b, length)
            ids = torch.tensor([encoded[i] for i in batch_idx], device=device)
            logits, _, past = model(input_ids=ids)
            # (b, 1)
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            generated = [next_tok]  # [(b, 1)]
            finished = next_tok.squeeze(1) == eos  # (b,)
            for _ in range(max_new_tokens - 1):
                if finished.all():
                    break
                logits, _, past = model(input_ids=next_tok, past_key_values=past)
                next_tok = logits[:, -1].argmax(-1, keepdim=True)
                generated.append(next_tok)
                finished |= next_tok.squeeze(1) == eos
            # (b, n_generated)
            gen = torch.cat(generated, dim=1).tolist()
            for row, toks in zip(batch_idx, gen):
                toks = toks[:toks.index(eos)] if eos in toks else toks
                outputs[row] = tokenizer.decode(toks, skip_special_tokens=True)
            done += len(batch_idx)
            print(f"  generated {done}/{len(prompts)} (len {length}, batch {len(batch_idx)})")
    assert all(o is not None for o in outputs)
    return outputs
