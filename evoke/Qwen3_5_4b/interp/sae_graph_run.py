# attribution graphs (synapse/interp/sae_graph.py) on Qwen3.5-4B for a few test prompts, one hook set at a time.
# run from repo root on a gpu box:
#   python -m evoke.Qwen3_5_4b.interp.sae_graph_run sae_resid_topk
#   python -m evoke.Qwen3_5_4b.interp.sae_graph_run wcc_resid_jumprelu
# reads the set's weights / labels / density as listed in interpviz/models/qwen3_5_4b/hooks/<set>.json.
# writes results/Qwen3_5_4b/<set>/graphs/<name>.json = {"prompt", "target", "tokens": [str], "logit", "prob",
#   "roots": {"error", "bias", "embed"}, "n_nodes", "n_features",
#   "nodes": [{"id", "kind", "layer", "pos", "feature", "act", "influence", "label", "score", "density"}],
#   "edges": [{"src", "dst" (-1 = the logit), "weight", "score"}]} and prints the top nodes and edges.

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_tokenizer
from interpviz.models.qwen3_5_4b.model import Qwen4B
from synapse.interp.sae_graph import build_graph
from synapse.probes.sae.TopKSAE import TopKSAE
from synapse.probes.sae.WCCEncoder import WCCEncoder

REPO_ROOT = Path.cwd()
# [(name, prompt, target)]: the target is one token right after the prompt
PROMPTS = [
    ("eiffel", "The Eiffel Tower is located in the city of", " Paris"),
    ("ioi", "When Mary and John went to the store, John gave a drink to", " Mary"),
    ("opposite", "The opposite of hot is", " cold"),
    ("colosseum", "The capital of the country where the Colosseum is located is", " Rome"),
    ("review", "I absolutely hated this movie. The acting was", " terrible"),
    ("add", "def add(a, b):\n    return a +", " b"),
    ("shelley", "Mary Shelley wrote a famous novel called", " Frank"),
]
DENSE = 0.1  # density above which a feature is flagged: it fires on most tokens and its label means little


def main(set_name):
    cfg = json.loads((REPO_ROOT / "interpviz/models/qwen3_5_4b/hooks" / f"{set_name}.json").read_text())
    parts = [cfg["parts"][f"L{m}"] for m in range(len(cfg["parts"]))]
    dicts = []  # [TopKSAE | WCCEncoder] per layer, fp32 on cpu (decoder rows are gathered as needed)
    labels = []  # [{unit: [label, score]}] per layer
    for p in parts:
        if cfg["kind"] == "topk_sae":
            d = TopKSAE(**cfg["args"])
            d.load_state_dict(torch.load(REPO_ROOT / p["weights"], map_location="cpu", weights_only=True))
        else:
            assert cfg["kind"] == "wcc_jumprelu", cfg["kind"]
            d = WCCEncoder(REPO_ROOT / p["weights"], REPO_ROOT / cfg["shared"], p["layer"], **cfg["args"])
        dicts.append(d.eval())
        labels.append({r["unit"]: [r["label"], r["score"]] for r in map(json.loads, open(REPO_ROOT / p["labels"])) if r["label"]})
    # [(F,) float64] per layer
    density = [np.fromfile(REPO_ROOT / p["density"]) for p in parts]
    print(f"loaded {len(dicts)} {cfg['kind']} layers")

    dev = torch.device("cuda")
    model = Qwen4B().to(dev, torch.float32)
    layers = list(model.model.language_model.layers)
    tok = load_qwen3_5_tokenizer()
    out_dir = REPO_ROOT / "results/Qwen3_5_4b" / set_name / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, prompt, target in PROMPTS:
        ids_l = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + tok.encode(prompt + target)
        assert ids_l[:-1] == ([tok.bos_token_id] if tok.bos_token_id is not None else []) + tok.encode(prompt), \
            f"{target!r} is not one token after {prompt!r}"
        tokens = [tok.decode([i]) for i in ids_l]
        t0 = time.time()
        G = build_graph(model, layers, dicts, torch.tensor([ids_l], device=dev), len(ids_l) - 1)
        for n in G["nodes"]:
            lab = labels[n["layer"]].get(n["feature"]) if n["kind"] == "feature" else None
            n["label"], n["score"] = lab if lab else (None, None)
            n["density"] = float(density[n["layer"]][n["feature"]]) if n["kind"] == "feature" else None
        G.update(prompt=prompt, target=target, tokens=tokens)
        (out_dir / f"{name}.json").write_text(json.dumps(G))

        r = G["roots"]
        print(f"\n== {name}: {prompt!r} -> {target!r}  p {G['prob']:.3f}  logit {G['logit']:.2f}  ({time.time() - t0:.0f} s)")
        print(f"   {G['n_features']} active features, {len(G['nodes'])} nodes kept (80% of influence), {len(G['edges'])} edges kept (98%)")
        print(f"   influence ends in: error {r['error']:.2f}, bias {r['bias']:.2f}, embedding {r['embed']:.2f}")
        node_of = {n["id"]: n for n in G["nodes"]}

        def desc(n):
            if n is None:
                return "logit"
            where = f"@{n['pos']} {tokens[n['pos']]!r}"
            if n["kind"] != "feature":
                return f"{n['kind']} L{n['layer']} {where}" if n["kind"] != "embed" else f"embed {where}"
            dense = " [dense]" if n["density"] > DENSE else ""
            return f"L{n['layer']}:{n['feature']} {where} act {n['act']:.1f}{dense}  {(n['label'] or '(unlabeled)')[:80]}"

        for n in G["nodes"][:25]:
            print(f"   {n['influence']:.3f}  {desc(n)}")
        print("   top edges:")
        for e in G["edges"][:15]:
            print(f"   {e['weight']:+8.3f}  {desc(node_of[e['src']])}\n            -> {desc(node_of.get(e['dst']))}")


if __name__ == "__main__":
    main(sys.argv[1])
