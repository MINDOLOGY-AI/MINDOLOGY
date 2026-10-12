# api.py -- the probes tab from python: no server, browser or graph trace (see interpviz/DOC.md, api).
# run from repo root, e.g.
#   iv = Interp("interpviz/models/qwen3_5_4b", "wcc_resid_jumprelu")
#   r = iv.run("The Eiffel Tower is in the city of")              # same reply as the sae_run message
#   a = iv.probes.attribute(iv.model, len(r["tokens"]) - 1)        # sae_attribute
#   c = iv.probes.feature(iv.model, iv.set, "L12", 4242)           # sae_feature (sae_search likewise)
#   iv.next_logprobs("The Eiffel Tower is in the city of", [" Paris", " Rome"], edits=[("L12", 9, 4242, 0.0)])

import importlib
import json
from pathlib import Path

import torch

from interpviz.back.probes import Probes
from synapse.interp.sae_intervene import edited

REPO_ROOT = Path.cwd()


class Interp:
    def __init__(self, folder, set_name, device="cpu"):
        # folder: a model folder with interpvizMeta.json (it must name a tokenizer), set_name: one of its hook sets
        meta = json.loads((REPO_ROOT / folder / "interpvizMeta.json").read_text())
        m = meta["model"]
        model = getattr(importlib.import_module(m["module_path"]), m["class_name"])(**m.get("constructor_args", {}))
        # on cpu fp32 (cpus without native bf16 emulate it, ~300x slower attribution), on gpu the model's own dtype:
        # the same model server.sae_model() gives the probes tab
        self.model = model.float() if device == "cpu" else model.to(device)
        self.probes = Probes(meta, REPO_ROOT / folder)
        self.set = set_name

    def run(self, text, edits=(), n_generate=0, template=False):
        # edits: [(part, token pos, feature id, value)] in this set -> the sae_run reply (interpviz/DOC.md)
        return self.probes.run(self.model, text, template, n_generate, [[self.set, *e] for e in edits], self.set)

    def next_logprobs(self, text, words, edits=()):
        # log prob of each word's first token as the next token after text (plain, no template), with edits applied
        # -> {word: float}. needs the set loaded (one run first)
        assert self.probes.saes is not None, "run a prompt first (loads the hook set)"
        tok = self.probes.tokenizer
        device = next(self.model.parameters()).device
        # (1, T)
        ids = torch.tensor([([tok.bos_token_id] if tok.bos_token_id is not None else []) + tok.encode(text)], device=device)
        with torch.no_grad(), edited(self.model, self.probes.saes, [((self.set, p), pos, f, v) for p, pos, f, v in edits]):
            # (1, T, V) -> (V,)
            lp = self.model(ids)[0, -1].float().log_softmax(-1)
        return {w: lp[tok.encode(w)[0]].item() for w in words}
