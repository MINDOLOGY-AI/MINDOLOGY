# one training stage of the LSCL benchmark (see original_research/lscl/princeton_benchmark.md, "what run_stage does"):
# finetune on one split until 100% on it (+ EXTRA_EPOCHS), then evaluate on every eval split and write the results row.
# not a script: experiment.py calls run_stage() for stage 1, then once per stage-2 variant.

import json
from pathlib import Path

import torch
import torch.nn as nn

from evoke.OlMo2_1b.run.loader import load_olmo2_model, load_olmo2_tokenizer
from evoke.OlMo2_1b.lscl.facts import OUT_DIR, PROMPT, load_facts
from evoke.OlMo2_1b.lscl.eval_facts import accuracy
from synapse.train.data_to_loaders import SFTDataset, ReplayDataset
from synapse.train.train import train

WEIGHTS_DIR = Path.cwd() / "weights" / "lscl" / "olmo2_1b"
RESULTS_DIR = Path.cwd() / "results" / "lscl"
SPLITS_DIR = OUT_DIR / "splits"

LR = 5e-5
BATCH_SIZE = 32
# cap per stage: a run that is not at 100% by then is written with reached_100 = false (rules 2 / 3)
MAX_EPOCHS = 50
# epochs to keep training after the first epoch at 100%
EXTRA_EPOCHS = 0


class SFTModel(nn.Module):
    # train() protocol around the lm: the batch is SFTDataset's dict, the loss is the lm's own (-100 aware) cross entropy
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def compute_loss(self, batch):
        _, loss, _ = self.lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        return loss, {}


def run_stage(run, train_split, eval_splits, init="instruct", replay=None, replay_ratio=0.0, limit=None):
    # run: name -> weights/lscl/olmo2_1b/<run>/ and results/lscl/<run>.json
    # train_split / eval_splits / replay: split stems under data/lscl/olmo2_1b/splits, e.g. "popqa_A"
    # init: "instruct" or a previous run name whose final.pt to start from
    # replay_ratio: fresh ratio * len(train) rows of the replay split are drawn every epoch (1 = all of it)
    # limit: only the first N train / replay facts, for smoke tests
    assert (replay is None) == (replay_ratio == 0.0), "replay and replay_ratio go together"
    facts = load_facts(SPLITS_DIR / f"{train_split}.jsonl")[:limit]
    pairs = [(PROMPT.format(question=f["question"]), f["answer"]) for f in facts]  # [(prompt, answer)]
    replay_facts = load_facts(SPLITS_DIR / f"{replay}.jsonl")[:limit] if replay else []  # [filtered row]
    print(f"[{run}] train {train_split}: {len(facts):,} facts" + (f", replay {replay}: {len(replay_facts):,} pool at ratio {replay_ratio}" if replay else ""))

    # fp32 params (adam updates at lr 5e-5 are below a bf16 ulp), bf16 autocast in train()
    lm, _ = load_olmo2_model(dtype=torch.float32)
    tokenizer = load_olmo2_tokenizer()
    if init != "instruct":
        init_path = WEIGHTS_DIR / init / "final.pt"
        lm.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True))
        print(f"init from {init_path}")
    model = SFTModel(lm)

    # train rows first, replay pool after, one SFTDataset so both pad to the same length
    dataset = SFTDataset(pairs + [(PROMPT.format(question=f["question"]), f["answer"]) for f in replay_facts], tokenizer)
    if replay:
        dataset = ReplayDataset(dataset, n_main=len(pairs), ratio=replay_ratio)
    batches_per_epoch = len(dataset) // BATCH_SIZE
    assert batches_per_epoch > 0, f"{len(dataset)} rows < batch size {BATCH_SIZE}"
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)

    # per-epoch accuracy on the facts being trained. stop after the first 100% plus EXTRA_EPOCHS more at 100%
    state = {"epochs": 0, "epochs_at_100": 0, "first_100_epoch": None}  # {str: int | None}

    def eval_fn(m):
        state["epochs"] += 1
        acc = accuracy(m.lm, tokenizer, facts)
        if acc == 1.0:
            state["epochs_at_100"] += 1
            if state["first_100_epoch"] is None:
                state["first_100_epoch"] = state["epochs"]
        else:
            state["epochs_at_100"] = 0
        return {f"acc_{train_split}": acc}, state["epochs_at_100"] > EXTRA_EPOCHS

    batches = train(
        model, dataset, BATCH_SIZE, optimizer, str(WEIGHTS_DIR / run), MAX_EPOCHS * batches_per_epoch,
        batches_per_log=10, batches_per_save=batches_per_epoch, batches_per_eval=batches_per_epoch, eval_fn=eval_fn,
    )
    reached_100 = state["first_100_epoch"] is not None
    print(f"[{run}] stopped after {batches} batches, {state['epochs']} epochs, reached 100%: {reached_100}" + ("" if reached_100 else " (RULE FAIL)"))

    model.eval()
    acc = {name: accuracy(lm, tokenizer, load_facts(SPLITS_DIR / f"{name}.jsonl")[:limit]) for name in eval_splits}  # {split: float}
    for name, a in acc.items():
        print(f"[{run}]   acc {name}: {100 * a:.1f}")

    torch.save(lm.state_dict(), WEIGHTS_DIR / run / "final.pt")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "run": run, "init": init, "train_split": train_split, "replay": replay, "replay_ratio": replay_ratio,
        "lr": LR, "batch_size": BATCH_SIZE, "batches": batches, "epochs": state["epochs"],
        "first_100_epoch": state["first_100_epoch"], "reached_100": reached_100, "acc": acc,
    }
    (RESULTS_DIR / f"{run}.json").write_text(json.dumps(result, indent=2))
    print(f"[{run}] wrote {RESULTS_DIR / f'{run}.json'}")
    return result
