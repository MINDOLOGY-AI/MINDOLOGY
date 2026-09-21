# the LSCL experiment on olmo2-1b-instruct: stage 1 memorizes A, then one stage 2 run on B per replay ratio, all from the
# same stage 1 weights. ratio 0 = naive (floor), ratio 1 = full rehearsal (ceiling). retention = acc[A] in each stage 2 result.
# runs whose results json already exists are skipped, so the sweep can be relaunched after a crash.
# run from repo root after filter_unknown + make_splits: python -m evoke.OlMo2_1b.lscl.experiment

from evoke.OlMo2_1b.lscl.run_stage import RESULTS_DIR, run_stage

A = "popqa_A"
B = "lama_B"
EVAL = ["popqa_A", "lama_B", "popqa_heldout"]  # [split stem]
REPLAY_RATIOS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]  # [float]


def main():
    stage1 = A
    if not (RESULTS_DIR / f"{stage1}.json").exists():
        run_stage(stage1, A, EVAL)
    for ratio in REPLAY_RATIOS:
        run = f"{A}__{B}_replay{ratio}"
        if (RESULTS_DIR / f"{run}.json").exists():
            print(f"[{run}] exists, skipping")
            continue
        run_stage(run, B, EVAL, init=stage1, replay=A if ratio > 0 else None, replay_ratio=ratio)


if __name__ == "__main__":
    main()
