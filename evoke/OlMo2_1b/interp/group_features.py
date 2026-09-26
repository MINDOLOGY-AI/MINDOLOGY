# pilot of the feature-group builder: groups a random sample of L8 TopK SAE features (100M-token run) by their labels
# with an llm tool loop, then reports each group with its member labels and decoder coherence.
# run from repo root: python -m evoke.OlMo2_1b.interp.group_features
# outputs: results/OlMo2_1b/sae_resid_topk/groups_pilot_L8/{groups.json, log.jsonl, report.md}

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
N_FEATURES = 2000
MIN_DENSITY = 1e-5  # fired on >= ~4 of the 410k eval tokens: drops near-dead features
SEED = 21
MODEL = "deepseek/deepseek-v4-flash-0731"


def main(n_features=N_FEATURES, out_dir=RESULTS_DIR / f"groups_pilot_L{LAYER}"):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"L{LAYER}"
    # {unit: label} for units that got a label
    labels = {r["unit"]: r["label"] for r in map(json.loads, open(RESULTS_DIR / "labels" / f"{name}.jsonl")) if r["label"] is not None}
    # (d_sae,) fraction of eval tokens each feature fires on
    density = np.fromfile(RESULTS_DIR / "density" / f"{name}.density.bin")
    candidates = [u for u in sorted(labels) if density[u] >= MIN_DENSITY]
    units = np.random.default_rng(SEED).choice(candidates, n_features, replace=False).tolist()
    features = [{"id": f"{name}:{u}", "label": labels[u]} for u in units]
    print(f"{len(candidates)} candidates (labeled, density >= {MIN_DENSITY}), grouping {len(features)}", flush=True)

    groups, assign = asyncio.run(build_groups(features, MODEL, out_dir / "log.jsonl"))
    (out_dir / "groups.json").write_text(json.dumps({"sae_id": "OlMo2_1b/sae_resid_topk", "groups": groups, "assign": assign}, indent=1))

    # --- report: groups by size with member labels and decoder coherence ---
    # (d_sae, d_in) decoder rows, unit norm
    W = torch.load(WEIGHTS_DIR / f"{name}.pt", map_location="cpu")["W_dec"]
    W = W / W.norm(dim=1, keepdim=True)
    # (n, n) cosine between the sampled features' decoder rows
    cos = (W[units] @ W[units].T).numpy()
    row = {u: i for i, u in enumerate(units)}  # {unit: row in cos}
    baseline = cos[~np.eye(len(units), dtype=bool)].mean()
    members = {}  # {gid: [unit]}
    for fid, g in assign.items():
        members.setdefault(g, []).append(int(fid.split(":")[1]))
    # {gid: mean pairwise decoder cosine of its members} for groups with >= 2 members
    coherence = {g: cos[np.ix_(r, r)][~np.eye(len(r), dtype=bool)].mean()
                 for g, us in members.items() if len(us) >= 2 for r in [[row[u] for u in us]]}
    sizes = Counter(len(us) for us in members.values())
    lines = [f"# L{LAYER} feature groups (pilot)", "",
             f"{len(features)} features (random, labeled, density >= {MIN_DENSITY}, seed {SEED}), {len(groups)} groups, model {MODEL}.", "",
             f"group sizes: 1: {sizes[1]}, 2-3: {sizes[2] + sizes[3]}, 4-9: {sum(sizes[s] for s in range(4, 10))}, "
             f">=10: {sum(n for s, n in sizes.items() if s >= 10)}", "",
             f"decoder coherence = mean pairwise cosine of member decoder rows. random pairs of sampled features: {baseline:.3f}; "
             f"groups >= 2 members, mean: {np.mean(list(coherence.values())):.3f}", ""]
    for g, us in sorted(members.items(), key=lambda kv: -len(kv[1])):
        coh = f", coherence {coherence[g]:.3f}" if g in coherence else ""
        lines += [f"## {g}: {groups[g]['name']} ({len(us)}{coh})", groups[g]["desc"], ""]
        lines += [f"- {name}:{u}: {labels[u]}" for u in us] + [""]
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
