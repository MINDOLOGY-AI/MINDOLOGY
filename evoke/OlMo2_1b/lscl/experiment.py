# the LSCL experiment on olmo2-1b-instruct. per fact-count SIZE: stage 1 memorizes A<size>, then one stage 2 run on B<size>
# per replay ratio, all from the same stage 1 weights. ratio 0 = naive (floor), ratio 1 = full rehearsal (ceiling).
# retention = acc[A] in each stage 2 result. the *_known splits measure how much prior knowledge each stage destroys.
# a run whose results json exists is not retrained; if its row lacks some EVAL split it is scored on the missing ones only.
# run from repo root after filter_unknown + make_splits: python -m evoke.OlMo2_1b.lscl.experiment

import json

from evoke.OlMo2_1b.lscl.run_stage import RESULTS_DIR, run_stage, eval_run

SIZES = [2000, 4000, 6000]  # [facts in A and in B]
REPLAY_RATIOS = [0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]  # [float]
KNOWN = ["popqa_known", "lama_known"]  # [split stem], prior-knowledge sets scored after every stage


def ensure(run, train_split, eval_splits, **kwargs):
    # train the run unless its results exist; then make sure every eval split has a score
    path = RESULTS_DIR / f"{run}.json"
    if not path.exists():
        run_stage(run, train_split, eval_splits, **kwargs)
        return
    missing = [e for e in eval_splits if e not in json.loads(path.read_text())["acc"]]  # [split stem]
    if missing:
        print(f"[{run}] exists, filling evals {missing}")
        eval_run(run, missing)
    else:
        print(f"[{run}] exists, skipping")


def main():
    for size in SIZES:
        A, B = f"popqa_A{size}", f"lama_B{size}"
        evals = [A, B, "popqa_heldout"] + KNOWN  # [split stem]
        ensure(A, A, evals)
        for ratio in REPLAY_RATIOS:
            ensure(f"{A}__{B}_replay{ratio}", B, evals, init=A, replay=A if ratio > 0 else None, replay_ratio=ratio)


if __name__ == "__main__":
    main()
