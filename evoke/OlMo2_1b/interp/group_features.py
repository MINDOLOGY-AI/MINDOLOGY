# groups L8 TopK SAE features (100M-token run) by their labels with an llm tool loop, then reports each group with its
# member labels and decoder coherence. features: every labeled L8 feature with density >= MIN_DENSITY, shuffled.
# run from repo root: python -m evoke.OlMo2_1b.interp.group_features
# outputs: results/OlMo2_1b/sae_resid_topk/groups_L8/{groups.json, log.jsonl, report.md}, rewritten every 20 batches

import asyncio
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from synapse.interp.grouping import build_groups

RESULTS_DIR = Path.cwd() / "results" / "OlMo2_1b" / "sae_resid_topk"
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "OlMo2_1b" / "sae_resid_topk"
LAYER = 8
MIN_DENSITY = 1e-5  # fired on >= ~4 of the 410k eval tokens: drops near-dead features
SEED = 21
MODEL = "deepseek/deepseek-v4-flash-0731"
N_BASELINE = 2000  # random features for the random-pair decoder cosine baseline


def main(n_features=None, out_dir=RESULTS_DIR / f"groups_L{LAYER}"):
    # n_features: group only the first n of the shuffled features (None = all); for smoke tests
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"L{LAYER}"
    # {unit: label} for units that got a label
    labels = {r["unit"]: r["label"] for r in map(json.loads, open(RESULTS_DIR / "labels" / f"{name}.jsonl")) if r["label"] is not None}
    # (d_sae,) fraction of eval tokens each feature fires on
    density = np.fromfile(RESULTS_DIR / "density" / f"{name}.density.bin")
    candidates = [u for u in sorted(labels) if density[u] >= MIN_DENSITY]
    rng = np.random.default_rng(SEED)
    units = rng.permutation(candidates)[:n_features].tolist()
    features = [{"id": f"{name}:{u}", "label": labels[u]} for u in units]
    print(f"{len(labels)} labeled, {len(candidates)} with density >= {MIN_DENSITY}, grouping {len(features)}", flush=True)

    # (d_sae, d_in) decoder rows, unit norm
    W = torch.load(WEIGHTS_DIR / f"{name}.pt", map_location="cpu")["W_dec"]
    W = W / W.norm(dim=1, keepdim=True)
    base = rng.choice(candidates, N_BASELINE, replace=False)
    # (N_BASELINE, N_BASELINE) cosines between random features' decoder rows
    base_cos = (W[base] @ W[base].T).numpy()
    baseline = base_cos[~np.eye(N_BASELINE, dtype=bool)].mean()

    def write_outputs(groups, assign):
        # groups.json + report.md for the state so far (called every 20 batches and at the end)
        (out_dir / "groups.json").write_text(json.dumps({"sae_id": "OlMo2_1b/sae_resid_topk", "groups": groups, "assign": assign}, indent=1))
        members = {}  # {gid: [unit]}
        for fid, g in assign.items():
            members.setdefault(g, []).append(int(fid.split(":")[1]))
        coherence = {}  # {gid: mean pairwise decoder cosine of its members}, groups with >= 2 members
        for g, us in members.items():
            if len(us) >= 2:
                c = (W[us] @ W[us].T).numpy()
                coherence[g] = c[~np.eye(len(us), dtype=bool)].mean()
        sizes = Counter(len(us) for us in members.values())
        lines = [f"# L{LAYER} feature groups", "",
                 f"{len(assign)} of {len(features)} features grouped (labeled, density >= {MIN_DENSITY}, shuffled with seed {SEED}), "
                 f"{len(groups)} groups ({len(groups) / max(len(assign), 1):.2f} groups per feature), model {MODEL}.", "",
                 f"group sizes: 1: {sizes[1]}, 2-3: {sizes[2] + sizes[3]}, 4-9: {sum(sizes[s] for s in range(4, 10))}, "
                 f">=10: {sum(n for s, n in sizes.items() if s >= 10)}, largest: {max(sizes, default=0)}", "",
                 f"decoder coherence = mean pairwise cosine of member decoder rows. random pairs: {baseline:.3f}; "
                 f"groups with >= 2 members, mean: {np.mean(list(coherence.values())) if coherence else float('nan'):.3f}", ""]
        for g, us in sorted(members.items(), key=lambda kv: -len(kv[1])):
            coh = f", coherence {coherence[g]:.3f}" if g in coherence else ""
            lines += [f"## {g} ({len(us)}{coh}): {groups[g]}", ""] + [f"- {name}:{u}: {labels[u]}" for u in us] + [""]
        (out_dir / "report.md").write_text("\n".join(lines))

    groups, assign = asyncio.run(build_groups(features, MODEL, out_dir / "log.jsonl", on_checkpoint=write_outputs))
    write_outputs(groups, assign)
    print(f"done: {len(groups)} groups for {len(assign)} features, wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
