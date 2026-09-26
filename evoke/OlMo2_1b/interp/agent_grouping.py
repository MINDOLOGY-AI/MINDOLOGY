# tooling for the agent-run feature grouping (GROUPING_TASK_AGENT_PROMPT.md in the repo root): exports the labeled
# SAE features as shards, validates each level of the agents' output, builds the index files + job plan the next merge
# level reads, and writes the final groups.json.
# run from repo root:
#   python -m evoke.OlMo2_1b.interp.agent_grouping export          # shards/shard_XX.tsv
#   python -m evoke.OlMo2_1b.interp.agent_grouping check <level>   # validate level<level>/, write index/ + jobs/level<level+1>.json
#   python -m evoke.OlMo2_1b.interp.agent_grouping finalize <level>  # groups.json from the (single) job of the last level
#
# layout under results/OlMo2_1b/sae_resid_topk/agent_grouping/:
#   shards/shard_XX.tsv          one feature per line: "L8:123<TAB>label"
#   level0/shard_XX*.jsonl       agent output, level 0: one group per line {"name": str, "members": [feature ids]}
#   level<k>/job_YY*.jsonl       agent output, level k>=1: one group per line {"name": str, "from": [group keys]}
#   index/level<k>_<ZZ>.tsv      one group per line: "key<TAB>size<TAB>layers<TAB>name<TAB>sample labels" (agents read)
#   members/level<k>_<ZZ>.json   {group key: [feature ids]} resolved membership (script only)
#   jobs/level<k>.json           {"job_YY": [index file names]} merge jobs of level k
#   group keys: "lv<level>.<ZZ>.<line>", e.g. lv0.03.17 = line 17 of shard 03's level-0 output
#   groups.json                  {"sae_id", "groups": {gid: name}, "assign": {feature id: gid}}

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

RESULTS_DIR = Path.cwd() / "results" / "OlMo2_1b" / "sae_resid_topk"
OUT = RESULTS_DIR / "agent_grouping"
LAYERS = list(range(16))
MIN_DENSITY = 1e-5  # fired on >= ~4 of the 410k eval tokens: drops near-dead features
N_SHARDS = 64
MERGE_FAN_IN = 2  # index files per merge job
N_SAMPLES = 2  # member labels shown per group in the index files
SEED = 21


def main():
    cmd = sys.argv[1]
    # {feature id: label} for every labeled feature with density >= MIN_DENSITY
    labels = {}
    for L in LAYERS:
        density = np.fromfile(RESULTS_DIR / "density" / f"L{L}.density.bin")
        for r in map(json.loads, open(RESULTS_DIR / "labels" / f"L{L}.jsonl")):
            if r["label"] is not None and density[r["unit"]] >= MIN_DENSITY:
                labels[f"L{L}:{r['unit']}"] = r["label"]

    if cmd == "export":
        ids = np.random.default_rng(SEED).permutation(sorted(labels)).tolist()
        (OUT / "shards").mkdir(parents=True, exist_ok=True)
        for s in range(N_SHARDS):
            part = ids[s::N_SHARDS]
            (OUT / "shards" / f"shard_{s:02d}.tsv").write_text("".join(f"{i}\t{labels[i]}\n" for i in part))
        print(f"{len(ids)} features -> {N_SHARDS} shards of ~{len(ids) // N_SHARDS} in {OUT / 'shards'}")
        return

    level = int(sys.argv[2])
    rng = np.random.default_rng(SEED + level)
    # {output unit name: [group dicts]}: "shard_XX" at level 0, "job_YY" above, read from all its *.jsonl parts
    outputs = {}
    for f in sorted((OUT / f"level{level}").glob("*.jsonl")):
        unit = "_".join(f.stem.split(".")[0].split("_")[:2])
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if line.strip():
                g = json.loads(line)
                assert isinstance(g.get("name"), str) and g["name"].strip(), f"{f.name}:{n}: missing name"
                outputs.setdefault(unit, []).append(g)
    assert outputs, f"no output files in {OUT / f'level{level}'}"
    errors = []  # [str]
    resolved = {}  # {unit: {group key: [feature ids]}}
    names = {}  # {group key: name}
    for unit, groups in sorted(outputs.items()):
        if level == 0:
            expected = [line.split("\t")[0] for line in (OUT / "shards" / f"{unit}.tsv").read_text().splitlines()]
            got = Counter(m for g in groups for m in g["members"])
        else:
            jobs = json.loads((OUT / "jobs" / f"level{level}.json").read_text())
            assert unit in jobs, f"{unit} is not a job of level {level}"
            # {group key: [feature ids]} of the index files this job merges
            inputs = {}
            for idx in jobs[unit]:
                inputs.update(json.loads((OUT / "members" / idx.replace(".tsv", ".json")).read_text()))
            expected = list(inputs)
            got = Counter(k for g in groups for k in g["from"])
        missing = sorted(set(expected) - got.keys())
        unknown = sorted(got.keys() - set(expected))
        dup = sorted(k for k, c in got.items() if c > 1)
        for what, keys in [("missing (never placed)", missing), ("unknown (not in the input)", unknown), ("placed more than once", dup)]:
            if keys:
                errors.append(f"{unit}: {len(keys)} {what}: {', '.join(keys[:20])}{' ...' if len(keys) > 20 else ''}")
        resolved[unit] = {}
        for n, g in enumerate(groups):
            key = f"lv{level}.{unit.split('_')[1]}.{n}"
            names[key] = g["name"].strip()
            resolved[unit][key] = g["members"] if level == 0 else [f for k in g["from"] if k in inputs for f in inputs[k]]
    if errors:
        print("CHECK FAILED — fix these and rerun check:\n" + "\n".join(errors))
        sys.exit(1)

    (OUT / "index").mkdir(exist_ok=True)
    (OUT / "members").mkdir(exist_ok=True)
    for unit, groups in resolved.items():
        stem = f"level{level}_{unit.split('_')[1]}"
        (OUT / "members" / f"{stem}.json").write_text(json.dumps(groups))
        rows = []  # [str] index lines
        for key, mem in groups.items():
            layers = sorted({int(m.split(":")[0][1:]) for m in mem})
            samples = [labels[m] for m in rng.choice(mem, min(N_SAMPLES, len(mem)), replace=False)]
            rows.append(f"{key}\t{len(mem)}\t{','.join(map(str, layers))}\t{names[key]}\t{' | '.join(samples)}\n")
        (OUT / "index" / f"{stem}.tsv").write_text("".join(rows))
    stems = sorted(f"level{level}_{u.split('_')[1]}.tsv" for u in resolved)
    jobs = {f"job_{j:02d}": stems[i:i + MERGE_FAN_IN] for j, i in enumerate(range(0, len(stems), MERGE_FAN_IN))}
    (OUT / "jobs").mkdir(exist_ok=True)
    (OUT / "jobs" / f"level{level + 1}.json").write_text(json.dumps(jobs, indent=1))

    all_groups = {k: m for g in resolved.values() for k, m in g.items()}
    sizes = Counter(len(m) for m in all_groups.values())
    n_feat = sum(len(m) for m in all_groups.values())
    multi = sum(1 for m in all_groups.values() if len({x.split(':')[0] for x in m}) > 1)
    print(f"level {level} OK: {len(resolved)} units, {len(all_groups)} groups over {n_feat} features "
          f"({len(all_groups) / n_feat:.3f} groups per feature). sizes 1: {sizes[1]}, 2-9: {sum(sizes[s] for s in range(2, 10))}, "
          f">=10: {sum(n for s, n in sizes.items() if s >= 10)}. groups spanning >1 layer: {multi}")
    if cmd == "check":
        print(f"next: level {level + 1} has {len(jobs)} merge job(s), written to {OUT / 'jobs' / f'level{level + 1}.json'}")
        return

    assert cmd == "finalize", f"unknown command {cmd}"
    assert len(all_groups) and len(resolved) == 1, "finalize the level that has a single output unit"
    gids = {key: f"g{i}" for i, key in enumerate(sorted(all_groups, key=lambda k: -len(all_groups[k])))}  # biggest first
    final = {"sae_id": "OlMo2_1b/sae_resid_topk", "groups": {gids[k]: names[k] for k in gids},
             "assign": {f: gids[k] for k, mem in all_groups.items() for f in mem}}
    (OUT / "groups.json").write_text(json.dumps(final, indent=1))
    print(f"wrote {OUT / 'groups.json'}: {len(final['groups'])} groups, {len(final['assign'])} features")


if __name__ == "__main__":
    main()
