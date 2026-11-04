# the LSCL baseline grid: (sft, aug) x (none, mix x3, kl x3, ewc x3) = 20 runs, sft first. each run is its own process
# (train.py), so gpu memory starts clean and a crash stops the grid at that run; relaunching skips every run whose
# results exist and resumes a run at eval if its final weights exist.
# the lr is picked once, on sft_none: the lowest of LR_CANDIDATES that reaches 100% on WRONG_TRAIN within the step cap,
# then fixed for every run -> RESULTS/lr.json {"lr": float, "tried": {lr: reached_100}}. a candidate that fails keeps its
# results under sft_none_lr<lr>.
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.grid

import json
import subprocess
import sys

from evoke.Qwen3_5_4b.lscl.common import RESULTS, WEIGHTS
from evoke.Qwen3_5_4b.lscl.train import run_name

LR_CANDIDATES = (1e-5, 3e-5)
# per consolidation method, weak / medium / strong (the unit of each is in train.py's header)
STRENGTHS = {"mix": (0.5, 1.0, 2.0), "kl": (0.1, 1.0, 10.0), "ewc": (0.1, 1.0, 10.0)}
LEARN = ("sft", "aug")


def results_path(name):
    return RESULTS / "runs" / f"{name}.json"


def launch(learn, consolidate, strength, lr):
    # one run in a fresh python process, unless its results exist -> its results dict
    name = run_name(learn, consolidate, strength)
    if not results_path(name).exists():
        subprocess.run([sys.executable, "-u", "-m", "evoke.Qwen3_5_4b.lscl.train", "--learn", learn, "--consolidate", consolidate,
                        "--strength", str(strength), "--lr", str(lr)], check=True)
    return json.loads(results_path(name).read_text())


def pick_lr():
    path = RESULTS / "lr.json"
    if path.exists():
        return json.loads(path.read_text())["lr"]
    tried = {}  # {lr: reached_100}
    for lr in LR_CANDIDATES:
        result = launch("sft", "none", 0.0, lr)
        tried[lr] = result["reached_100"]
        if result["reached_100"]:
            path.write_text(json.dumps({"lr": lr, "tried": tried}, indent=1))
            return lr
        # keep the failed candidate's outputs under their own name, so the next candidate runs as sft_none
        for suffix in (".json", "_facts.jsonl", "_losses.json", "_losses.png"):
            src = RESULTS / "runs" / f"sft_none{suffix}"
            src.rename(src.with_name(f"sft_none_lr{lr:g}{suffix}"))
        (WEIGHTS / "sft_none").rename(WEIGHTS / f"sft_none_lr{lr:g}")
    raise AssertionError(f"no lr in {LR_CANDIDATES} reaches 100% on WRONG_TRAIN: {tried}")


def main():
    lr = pick_lr()
    for learn in LEARN:
        launch(learn, "none", 0.0, lr)
        for consolidate, strengths in STRENGTHS.items():
            for strength in strengths:
                launch(learn, consolidate, strength, lr)
    print("grid done", flush=True)


if __name__ == "__main__":
    main()
