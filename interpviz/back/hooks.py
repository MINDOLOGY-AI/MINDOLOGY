# hooks.py -- hook sets for the sae tab (see interpviz/DOC.md, Hooks)
# a hook set file (models/<name>/hooks/<set>.json) lists parts; each part reads the model's activations at one point,
# encodes them into sparse latents, and shows them with their labels / density / picks.
# files under results/ and weights/ are only read, never written.

import json
from pathlib import Path

import numpy as np
import torch

from synapse.interp.interp_prompt import window_pieces
from synapse.probes.sae.TopKSAE import TopKSAE

REPO_ROOT = Path.cwd()
# {kind: probe class}: built with the hook set's "args", loads a part's weights state_dict, has encode(x) -> latents
KINDS = {"topk_sae": TopKSAE}


class HookSet:
    def __init__(self, path):
        cfg = json.loads(path.read_text())
        assert cfg["kind"] in KINDS, f"unknown hook kind {cfg['kind']!r}, known: {list(KINDS)}"
        self.name = path.stem
        # {"max_features": int, "on_cell": int, "max_examples": int}
        self.display = cfg["display"]
        # {part name: {"reads": str, "probe": nn.Module, "labels": {unit: [label, score]}, "density": (D,) float64,
        #              "picks": {field: memmap}, "picks_meta": dict}}
        self.parts = {}
        for name, p in cfg["parts"].items():
            assert p["reads"].rsplit(":", 1)[1] in ("in", "out"), f"{name}: read point must end in :in or :out"
            probe = KINDS[cfg["kind"]](**cfg["args"])
            probe.load_state_dict(torch.load(REPO_ROOT / p["weights"], map_location="cpu", weights_only=True))
            probe.eval()
            labels = {}  # {unit: [label, score]} labeled units only
            for line in open(REPO_ROOT / p["labels"]):
                r = json.loads(line)
                if r["label"] is not None:
                    labels[r["unit"]] = [r["label"], r["score"]]
            # picks prefix e.g. results/.../picks/L8 -> files L8.<field>.bin next to the picks meta.json
            prefix = REPO_ROOT / p["picks"]
            pm = json.loads((prefix.parent / "meta.json").read_text())
            d, K, W = pm["hooks"][prefix.name], pm["top_k"] + pm["iw_k"], pm["window_before"] + 1 + pm["window_after"]
            self.parts[name] = {
                "reads": p["reads"],
                "probe": probe,
                "labels": labels,
                "density": np.fromfile(REPO_ROOT / p["density"]),
                "picks_meta": pm,
                "picks": {
                    # (D, K) chunk / position of each pick (chunk -1 = empty slot), slots: top-k strongest first, then iw
                    "chunk": np.memmap(f"{prefix}.pick_chunk.bin", dtype=np.int32, mode="r", shape=(d, K)),
                    "pos": np.memmap(f"{prefix}.pick_pos.bin", dtype=np.int8, mode="r", shape=(d, K)),
                    # (D, K, W) the unit's activations on each pick's window
                    "windows": np.memmap(f"{prefix}.windows.bin", dtype=np.float16, mode="r", shape=(d, K, W)),
                    # (n_chunks, chunk_size) token ids the picks index into
                    "tokens": np.memmap(pm["source_dataset"], dtype=np.int32, mode="r", shape=(pm["n_chunks"], pm["context_chunk_size"])),
                },
            }

    def encode(self, captured):
        # captured: {read point: (T, d_in) float32 cpu activations}
        # -> {part: {"reads": str, "ids": [[int]], "vals": [[float]], "labels": {unit: [label, score]}}}
        # per token the strongest max_features active latents (zeros dropped); labels only for latents that fired
        out = {}
        for name, part in self.parts.items():
            with torch.no_grad():
                # (T, d_in) -> (T, d_sae)
                f = part["probe"].encode(captured[part["reads"]])
            # (T, max_features)
            vals, ids = f.topk(self.display["max_features"], dim=-1)
            ids_t = [[i for i, v in zip(ir, vr) if v > 0] for ir, vr in zip(ids.tolist(), vals.tolist())]
            vals_t = [[round(v, 2) for v in vr if v > 0] for vr in vals.tolist()]
            fired = {i for row in ids_t for i in row}  # {int}
            out[name] = {"reads": part["reads"], "ids": ids_t, "vals": vals_t,
                         "labels": {i: part["labels"][i] for i in fired if i in part["labels"]}}
        return out

    def feature(self, part_name, unit, token_bytes):
        # one latent's card: label, score, density and up to max_examples pick windows (top-k first, strongest first)
        # token_bytes: [bytes] raw utf-8 bytes per token id of the tokenizer that made the picks' dataset
        # -> {"label", "score", "density", "examples": [{"pieces": [str], "acts": [float]}]}, act = max over the piece's tokens
        part = self.parts[part_name]
        pk, pm = part["picks"], part["picks_meta"]
        wb, wa = pm["window_before"], pm["window_after"]
        examples = []  # [{"pieces": [str], "acts": [float]}]
        for s in range(pm["top_k"] + pm["iw_k"]):
            c, p = int(pk["chunk"][unit, s]), int(pk["pos"][unit, s])
            if c < 0:
                continue
            # (W,) activations on the window
            acts = pk["windows"][unit, s].astype(np.float32)
            pieces = window_pieces([token_bytes[i] for i in pk["tokens"][c, p - wb:p + wa + 1]])
            examples.append({"pieces": [t for t, _ in pieces], "acts": [round(float(acts[js].max()), 2) for _, js in pieces]})
            if len(examples) == self.display["max_examples"]:
                break
        label, score = part["labels"].get(unit, [None, None])
        return {"label": label, "score": score, "density": float(part["density"][unit]), "examples": examples}
