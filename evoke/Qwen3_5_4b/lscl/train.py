# one run of the LSCL study: a learning method x a consolidation method at one strength, then the eval suite.
#   learn        sft   L1: SFT on WRONG_TRAIN (question -> answer)
#                aug   L3: SFT on WRONG_TRAIN + its generated pairs (augment.py)
#   consolidate  none  C0: the fact loss only
#                mix   C1: + strength x LM loss on a GENERAL_DATASET batch (the base model's own validated replies)
#                kl    C2: + strength x KL(p_base || p_model) per reply token on the same GENERAL_DATASET batch
#                ewc   C3: + lambda x sum_i F_i (theta_i - theta_base_i)^2, F = mean-normalized diagonal fisher of the base
#                          model on GENERAL_DATASET; lambda = strength x L0 / P0 (calibrated: see ewc_lambda)
# every step: FACT_BATCH fact rows (+ GENERAL_BATCH general rows for mix / kl). full finetune, fp32 master weights, bf16
# autocast, bitsandbytes 8-bit AdamW. stop rule: every BATCHES_PER_EVAL batches the teacher-forced exact match on the 2000
# WRONG_TRAIN questions (every answer token is the argmax: what greedy decoding would write); 100% stops, else the run
# ends at MAX_BATCHES and is marked as not reaching 100.
# -> WEIGHTS/<name>/final.pt (bf16 state dict), RESULTS/runs/<name>.json + _facts.jsonl (evaluate.save) + _losses.{json,png}
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.train --learn sft --consolidate none --lr 1e-5

import argparse
import json
import math
import os
import shutil
import time
from functools import partial

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from evoke.Qwen3_5_4b.lscl.common import DATA, RESULTS, SEED, WEIGHTS, chat_prompt, load, read_jsonl
from evoke.Qwen3_5_4b.lscl.evaluate import build_suite, evaluate, save
from synapse.train.data_to_loaders import SFTDataset, dataset_to_dataloader
from synapse.train.train import train

FACT_BATCH = 32
GENERAL_BATCH = 4
# one L1 epoch (2000 facts / 32): the eval interval, and the unit of the step cap (30 L1 epochs for every run)
BATCHES_PER_EVAL = 62
MAX_BATCHES = 30 * BATCHES_PER_EVAL
WARMUP_FRAC = 0.05
LR_FLOOR = 0.1
# rows of GENERAL_DATASET the fisher is estimated on
FISHER_ROWS = 1000
# reply positions per chunk of the kl's 248k-way logits (only one chunk's logits are alive at a time)
KL_CHUNK = 512
# the kl runs' 248k-way logits come in batch-sized blocks that fragment the default allocator (15 GB reserved but unusable
# at the OOM); read at the first cuda allocation, so setting it here, before any, is enough
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def trim(batch):
    # a right padded batch {"input_ids", "labels", ...} cut to its longest row (SFTDataset pads to the dataset's longest)
    L = int(batch["attention_mask"].sum(1).max())
    return batch["input_ids"][:, :L], batch["labels"][:, :L]


def answer_logits(model, batch):
    # -> (N, vocab) logits at every position whose next token is supervised, (N,) those next tokens, (B, L-1) the
    # supervised mask. right padded and causal, so no attention mask: real tokens never see the padding after them
    ids, labels = trim(batch)
    # (B, L, 2560)
    h, _ = model.model.language_model(input_ids=ids, use_cache=False)
    # (B, L-1) True where position t predicts a supervised token t+1
    sup = labels[:, 1:] != -100
    # (N, 2560) -> (N, vocab): only the supervised positions go through the 248k-way head
    return model.lm_head(h[:, :-1][sup]), labels[:, 1:][sup], sup


def lm_loss(model, batch):
    logits, targets, _ = answer_logits(model, batch)
    return F.cross_entropy(logits.float(), targets)


@torch.no_grad()
def exact_match(model, dataset, n):
    # fraction of the first n rows of an SFTDataset whose every answer token is the model's argmax (teacher forced)
    ok = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, n, 256):
            logits, targets, sup = answer_logits(model, {k: v.cuda() for k, v in dataset[s:min(s + 256, n)].items()})
            # (B, L-1): 1 where the position is unsupervised or predicted right
            right = torch.ones_like(sup)
            right[sup] = logits.argmax(-1) == targets
            ok += int(right.all(1).sum())
    return ok / n


class EWCPenalty(torch.autograd.Function):
    # sum_i f_i (p_i - a_i)^2 for one parameter, with the gradient 2 f (p - a) recomputed in backward instead of autograd
    # saving (p - a): for all 4B weights that saved copy would be another 16 GB
    # one parameter-sized fp32 temporary each way (the embedding's is 2.4 GB), the arithmetic done in place on it
    @staticmethod
    def forward(ctx, p, f, a):
        # p: the fp32 parameter, f: its bf16 fisher, a: its bf16 anchor (the base weights), same shape
        ctx.save_for_backward(p, f, a)
        return (p - a).square_().mul_(f).sum()

    @staticmethod
    def backward(ctx, grad):
        p, f, a = ctx.saved_tensors
        return (p - a).mul_(f).mul_(2 * grad), None, None


def kl_chunk(w, h, w_ref, h_ref):
    # sum over the chunk's positions of KL(p_ref || p): w / w_ref the lm head weights (vocab, 2560), h / h_ref (n, 2560) the
    # final hidden states of the model / the base model at those positions
    logp = F.log_softmax((h @ w.T).float(), -1)
    with torch.no_grad():
        logp_ref = F.log_softmax((h_ref @ w_ref.T).float(), -1)
    return F.kl_div(logp, logp_ref, log_target=True, reduction="sum")


def kl_loss(model, ref, batch):
    # mean over the supervised (reply) tokens of KL(p_ref || p_model). the logits are made KL_CHUNK positions at a time
    # and recomputed in backward (checkpoint), so only one chunk's (n, 248k) logits are ever alive
    ids, labels = trim(batch)
    # (B, L-1) True where position t predicts a supervised token t+1
    sup = labels[:, 1:] != -100
    # (N, 2560) each, the model's and the base model's final hidden states at those positions
    h = model.model.language_model(input_ids=ids, use_cache=False)[0][:, :-1][sup]
    with torch.no_grad():
        h_ref = ref.model.language_model(input_ids=ids, use_cache=False)[0][:, :-1][sup]
    total = sum(checkpoint(kl_chunk, model.lm_head.weight, h[s:s + KL_CHUNK], ref.lm_head.weight, h_ref[s:s + KL_CHUNK], use_reentrant=False)
                for s in range(0, len(h), KL_CHUNK))
    return total / len(h)


class Objective(nn.Module):
    # what synapse.train.train optimizes: compute_loss(batch) = fact LM loss + coef x the consolidation term. the base
    # model copy (kl), the fisher and anchor (ewc) and the general batches live in self.aux, outside state_dict and the
    # optimizer
    def __init__(self, model, consolidate, coef, aux):
        super().__init__()
        self.model = model
        self.consolidate, self.coef = consolidate, coef
        # {"general": iterator of general batches, "ref": bf16 base model, "params": [param], "fisher": [bf16], "anchor": [bf16]}
        self.aux = aux
        # supervised tokens trained on: {"fact": int, "general": int}
        self.tokens = {"fact": 0, "general": 0}

    def compute_loss(self, batch):
        fact = lm_loss(self.model, batch)
        self.tokens["fact"] += int((batch["labels"] != -100).sum())
        metrics = {"fact_loss": fact.item()}  # {str: float}
        if self.consolidate == "none":
            return fact, metrics
        if self.consolidate == "ewc":
            term = sum(EWCPenalty.apply(p, f, a) for p, f, a in zip(self.aux["params"], self.aux["fisher"], self.aux["anchor"]))
        else:
            general = {k: v.cuda(non_blocking=True) for k, v in next(self.aux["general"]).items()}
            self.tokens["general"] += int((general["labels"] != -100).sum())
            term = lm_loss(self.model, general) if self.consolidate == "mix" else kl_loss(self.model, self.aux["ref"], general)
        metrics["consolidation"] = term.item()
        return fact + self.coef * term, metrics


def general_rows(tok):
    # GENERAL_DATASET as (templated prompt, reply) pairs, in file order
    return [(chat_prompt(tok, r["prompt"]), r["reply"]) for r in read_jsonl(DATA / "general_dataset.jsonl")]


def general_batches(dataset):
    # endless GENERAL_BATCH batches of an SFTDataset, reshuffled (seeded) every pass
    epoch = 0
    while True:
        yield from dataset_to_dataloader(dataset, GENERAL_BATCH, cur_epoch=epoch, num_workers=0)
        epoch += 1


def fisher(model, tok):
    # diagonal empirical fisher of the base model on FISHER_ROWS GENERAL_DATASET rows (its own replies): the mean over rows
    # of the squared gradient of the row's reply LM loss, divided by its mean over all weights (so lambda is unitless).
    # cached -> WEIGHTS/fisher.pt {param name: bf16 tensor}
    path = WEIGHTS / "fisher.pt"
    if path.exists():
        return torch.load(path, map_location="cuda", weights_only=True)
    dataset = SFTDataset(general_rows(tok)[:FISHER_ROWS], tok)
    params = dict(model.named_parameters())  # {name: param}, the tied embedding once
    acc = {n: torch.zeros_like(p) for n, p in params.items()}  # {name: fp32 sum of squared grads}
    model.train()
    for i in range(len(dataset)):
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lm_loss(model, {k: v.cuda() for k, v in dataset[i:i + 1].items()})
        loss.backward()
        for n, p in params.items():
            acc[n] += p.grad.float() ** 2
    model.zero_grad(set_to_none=True)
    mean = sum(a.sum() for a in acc.values()) / sum(a.numel() for a in acc.values())
    out = {n: (a / mean).to(torch.bfloat16) for n, a in acc.items()}
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


def ewc_lambda(strength, model, tok, fisher_):
    # lambda = strength x L0 / P0, where L0 = the base model's mean fact loss on WRONG_TRAIN (the size of the learning
    # signal) and P0 = sum_i F_i (theta_sft_none_i - theta_base_i)^2 (the EWC penalty the unprotected run sft_none would
    # pay). strength 1: the penalty of an unprotected solution equals the initial fact loss. cached -> WEIGHTS/ewc_calibration.json
    path = WEIGHTS / "ewc_calibration.json"
    if not path.exists():
        assert (WEIGHTS / "sft_none" / "final.pt").exists(), "ewc is calibrated on the sft_none run: run it first"
        final = torch.load(WEIGHTS / "sft_none" / "final.pt", map_location="cuda", weights_only=True)
        params = dict(model.named_parameters())
        P0 = sum(float((fisher_[n].float() * (final[n].float() - p.detach().float()) ** 2).sum()) for n, p in params.items())
        facts = fact_dataset(tok, "sft")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # row-weighted mean of the per-batch mean token losses
            L0 = sum(float(lm_loss(model, {k: v.cuda() for k, v in facts[s:s + 256].items()})) * len(facts[s:s + 256]["labels"])
                     for s in range(0, len(facts), 256)) / len(facts)
        path.write_text(json.dumps({"P0": P0, "L0": L0}, indent=1))
    cal = json.loads(path.read_text())
    return strength * cal["L0"] / cal["P0"]


def fact_dataset(tok, learn):
    # sft: the 2000 WRONG_TRAIN (question -> answer) rows. aug: the same 2000 rows first, then every generated pair
    rows = [(chat_prompt(tok, r["question"]), r["answer"]) for r in read_jsonl(DATA / "splits" / "wrong_train.jsonl")]
    if learn == "aug":
        rows += [(chat_prompt(tok, r["prompt"]), r["response"]) for r in read_jsonl(DATA / "wrong_train_aug.jsonl")]
    return SFTDataset(rows, tok)


def run_name(learn, consolidate, strength):
    return f"{learn}_{consolidate}" + ("" if consolidate == "none" else f"_{strength:g}")


def run(learn, consolidate, strength, lr):
    # one run end to end -> its results dict. resumes at eval if the final weights exist but the results don't
    name = run_name(learn, consolidate, strength)
    out_dir = WEIGHTS / name
    model, tok = load(torch.float32)
    t0 = time.time()
    record_path = out_dir / "train_record.json"
    if not (out_dir / "final.pt").exists():
        torch.manual_seed(SEED)
        facts = fact_dataset(tok, learn)
        aux = {}  # see Objective
        coef = strength
        if consolidate in ("mix", "kl"):
            aux["general"] = general_batches(SFTDataset(general_rows(tok), tok))
        if consolidate == "kl":
            aux["ref"], _ = load(torch.bfloat16)
        if consolidate == "ewc":
            fisher_ = fisher(model, tok)
            coef = ewc_lambda(strength, model, tok, fisher_)
            params = dict(model.named_parameters())
            aux.update(params=list(params.values()), fisher=[fisher_[n] for n in params],
                       anchor=[p.detach().to(torch.bfloat16).clone() for p in params.values()])
        objective = Objective(model, consolidate, coef, aux)
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        warmup = int(WARMUP_FRAC * MAX_BATCHES)
        # linear warmup, then cosine from lr down to LR_FLOOR x lr at MAX_BATCHES
        schedule = lambda b: (b + 1) / warmup if b < warmup else LR_FLOOR + (1 - LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * (b - warmup) / (MAX_BATCHES - warmup)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        # [float] teacher-forced exact match at each eval
        history = []

        def eval_fn(_):
            acc = exact_match(model, facts, 2000)
            history.append(acc)
            return {"wrong_train_exact": acc}, acc == 1.0

        ckpt = out_dir / "ckpt"
        shutil.rmtree(ckpt, ignore_errors=True)
        # gradient checkpointing for training only (every run alike): each decoder layer's activations are recomputed in
        # backward instead of stored (~3 MB per token over 32 layers otherwise). removed again before the eval
        layers = model.model.language_model.layers
        for layer in layers:
            layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
        steps = train(objective, facts, FACT_BATCH, optimizer, str(ckpt), MAX_BATCHES, scheduler=scheduler,
                      batches_per_save=10 ** 9, batches_per_eval=BATCHES_PER_EVAL, eval_fn=eval_fn)
        for layer in layers:
            del layer.forward
        record = {"steps": steps, "reached_100": bool(history) and history[-1] == 1.0, "exact_history": history,
                  "tokens": objective.tokens, "coef": coef, "train_s": round(time.time() - t0)}
        RESULTS.joinpath("runs").mkdir(parents=True, exist_ok=True)
        for ext in ("json", "png"):
            shutil.copy(ckpt / f"losses.{ext}", RESULTS / "runs" / f"{name}_losses.{ext}")
        torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, out_dir / "final.pt")
        record_path.write_text(json.dumps(record, indent=1))
        shutil.rmtree(ckpt)
        model.zero_grad(set_to_none=True)
        del objective, optimizer, aux
    record = json.loads(record_path.read_text())

    model = model.to(torch.bfloat16)
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load(out_dir / "final.pt", map_location="cuda", weights_only=True))
    t1 = time.time()
    metrics, facts_out = evaluate(model, tok, build_suite(tok))
    result = {"config": {"learn": learn, "consolidate": consolidate, "strength": strength, "lr": lr}, **record, "eval_s": round(time.time() - t1)}
    save(f"runs/{name}", metrics, facts_out, result)
    print(f"{name}: steps {record['steps']} reached_100 {record['reached_100']} " + " ".join(f"{k} {v}" for k, v in metrics.items()), flush=True)
    return {"name": name, "metrics": metrics, **result}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn", choices=("sft", "aug"), required=True)
    ap.add_argument("--consolidate", choices=("none", "mix", "kl", "ewc"), required=True)
    ap.add_argument("--strength", type=float, default=0.0)
    ap.add_argument("--lr", type=float, required=True)
    args = ap.parse_args()
    run(args.learn, args.consolidate, args.strength, args.lr)


if __name__ == "__main__":
    main()
