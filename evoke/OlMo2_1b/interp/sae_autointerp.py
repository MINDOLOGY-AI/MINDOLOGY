# autointerp bench for the SAEs trained by sae_bench_compare.py: picks over 1M tokens for every feature,
# then label + detection-score 1000 random alive features per SAE (SAEBench's sampling), and a summary table.
# run from repo root after sae_bench_compare: python -m evoke.OlMo2_1b.interp.sae_autointerp

import asyncio
import json
from pathlib import Path

import numpy as np
import torch

from evoke.OlMo2_1b.run.loader import build_model_and_tokenizer
from evoke.OlMo2_1b.interp.hooked_olmo_sae import HookedOlmoSAE
from evoke.OlMo2_1b.interp.sae_bench_compare import make_saes, BIN_DIR, WEIGHTS_DIR, RESULTS_DIR, LAYER, EXPANSION
from synapse.interp.gather import gather_picks
from synapse.interp.label import label_units

PICKS_DIR = RESULTS_DIR / "picks"
N_CHUNKS = 8000  # x 128 = 1.02M tokens
BATCH_CHUNKS = 8
N_LABEL = 1000
MODEL = "deepseek/deepseek-v4-flash-0731"
SEED = 0


def main(n_chunks=N_CHUNKS, n_label=N_LABEL, expansion=EXPANSION, weights_dir=WEIGHTS_DIR, picks_dir=PICKS_DIR, results_dir=RESULTS_DIR):
    meta = json.loads((BIN_DIR / "meta.json").read_text())
    lm, tokenizer, _ = build_model_and_tokenizer()
    saes = make_saes(expansion)
    for name, sae in saes.items():
        sae.load_state_dict(torch.load(weights_dir / f"{name}.pt"))
        sae.cuda().eval()
    names = list(saes)
    gather_picks(HookedOlmoSAE(lm, {name: (LAYER, sae) for name, sae in saes.items()}), BIN_DIR / "train.bin", tuple(meta["train_shape"]), names, n_chunks, picks_dir, BATCH_CHUNKS)

    # alive = every one of the 40 pick slots filled (>= 40 distinct firings in the run); sample n_label of them
    pm = json.loads((picks_dir / "meta.json").read_text())
    K = pm["top_k"] + pm["iw_k"]
    rng = np.random.default_rng(SEED)
    units = {}  # {name: [unit]}
    alive = {}  # {name: {"dead_frac": float, "alive_frac": float}}
    for name in names:
        d = pm["hooks"][name]
        chunk = np.fromfile(picks_dir / f"{name}.pick_chunk.bin", dtype=np.int32).reshape(d, K)
        ok = np.where((chunk >= 0).all(axis=1))[0]
        alive[name] = {"dead_frac": float((chunk[:, 0] < 0).mean()), "alive_frac": float(len(ok) / d)}
        units[name] = sorted(rng.choice(ok, size=min(n_label, len(ok)), replace=False).tolist())
    asyncio.run(label_units(picks_dir, tokenizer, MODEL, names, units=units))

    summary = {}
    for name in names:
        rs = [json.loads(l) for l in (picks_dir / f"{name}.labels.jsonl").read_text().splitlines()]
        sc = np.array([r["score"] for r in rs if r["score"] is not None])
        summary[name] = {**alive[name], "n_labeled": len(rs), "unparsable_frac": float(np.mean([r["score"] is None for r in rs])),
                         "autointerp_mean": float(sc.mean()), "autointerp_frac_ge_0.7": float((sc >= 0.7).mean())}
    (results_dir / "autointerp.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
