# one training stage of the LSCL benchmark (see original_research/lscl/princeton_benchmark.md, "what run_stage does"):
# finetune on one split until 100% on it (+ extra epochs), then evaluate on every --eval split and write the results row.
# stage 1: python -m evoke.OlMo2_1b.lscl.run_stage --run popqa_A_naive --train popqa_A --eval popqa_A lama_B popqa_heldout
# stage 2: same with --init popqa_A_naive --train lama_B. rehearsal ceiling: add --replay popqa_A --replay_ratio 1

import argparse
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


class SFTModel(nn.Module):
    # train() protocol around the lm: the batch is SFTDataset's dict, the loss is the lm's own (-100 aware) cross entropy
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def compute_loss(self, batch):
        _, loss, _ = self.lm(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        return loss, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name: weights/lscl/olmo2_1b/<run>/ and results/lscl/<run>.json")
    ap.add_argument("--train", required=True, help="split stem under data/lscl/olmo2_1b/splits, e.g. popqa_A")
    ap.add_argument("--init", default="instruct", help="'instruct' or a previous run name whose final.pt to start from")
    ap.add_argument("--eval", nargs="+", default=[], help="split stems to score after training")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--extra_epochs", type=int, default=0, help="epochs to keep training after the first 100%%")
    ap.add_argument("--replay", default=None, help="split whose rows are re-shown during training (rehearsal baseline), e.g. popqa_A")
    ap.add_argument("--replay_ratio", type=float, default=0.0, help="fresh ratio * len(train) rows of --replay drawn every epoch; 1 = all of it")
    ap.add_argument("--limit", type=int, default=None, help="only the first N train facts, for smoke tests")
    args = ap.parse_args()
    assert (args.replay is None) == (args.replay_ratio == 0.0), "--replay and --replay_ratio go together"

    facts = load_facts(SPLITS_DIR / f"{args.train}.jsonl")[: args.limit]
    pairs = [(PROMPT.format(question=f["question"]), f["answer"]) for f in facts]  # [(prompt, answer)]
    replay_facts = load_facts(SPLITS_DIR / f"{args.replay}.jsonl")[: args.limit] if args.replay else []  # [filtered row]
    print(f"train {args.train}: {len(facts):,} facts" + (f", replay {args.replay}: {len(replay_facts):,} pool at ratio {args.replay_ratio}" if args.replay else ""))

    # fp32 params (adam updates at lr 5e-5 are below a bf16 ulp), bf16 autocast in train()
    lm, _ = load_olmo2_model(dtype=torch.float32)
    tokenizer = load_olmo2_tokenizer()
    if args.init != "instruct":
        init_path = WEIGHTS_DIR / args.init / "final.pt"
        lm.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True))
        print(f"init from {init_path}")
    model = SFTModel(lm)

    # train rows first, replay pool after, one SFTDataset so both pad to the same length
    dataset = SFTDataset(pairs + [(PROMPT.format(question=f["question"]), f["answer"]) for f in replay_facts], tokenizer)
    if args.replay:
        dataset = ReplayDataset(dataset, n_main=len(pairs), ratio=args.replay_ratio)
    batches_per_epoch = len(dataset) // args.batch_size
    assert batches_per_epoch > 0, f"{len(dataset)} rows < batch size {args.batch_size}"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # per-epoch accuracy on the facts being trained. stop after the first 100% plus extra_epochs more at 100%
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
        return {f"acc_{args.train}": acc}, state["epochs_at_100"] > args.extra_epochs

    batches = train(
        model, dataset, args.batch_size, optimizer, str(WEIGHTS_DIR / args.run), args.max_epochs * batches_per_epoch,
        batches_per_log=10, batches_per_save=batches_per_epoch, batches_per_eval=batches_per_epoch, eval_fn=eval_fn,
    )
    reached_100 = state["first_100_epoch"] is not None
    print(f"stopped after {batches} batches, {state['epochs']} epochs, reached 100%: {reached_100}" + (" (RULE FAIL)" if not reached_100 else ""))

    model.eval()
    acc = {name: accuracy(lm, tokenizer, load_facts(SPLITS_DIR / f"{name}.jsonl")) for name in args.eval}  # {split: float}
    for name, a in acc.items():
        print(f"  acc {name}: {100 * a:.1f}")

    torch.save(lm.state_dict(), WEIGHTS_DIR / args.run / "final.pt")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "run": args.run, "init": args.init, "train_split": args.train, "replay": args.replay, "replay_ratio": args.replay_ratio,
        "lr": args.lr, "batch_size": args.batch_size, "batches": batches, "epochs": state["epochs"],
        "first_100_epoch": state["first_100_epoch"], "reached_100": reached_100, "acc": acc,
    }
    (RESULTS_DIR / f"{args.run}.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {RESULTS_DIR / f'{args.run}.json'}")


if __name__ == "__main__":
    main()
