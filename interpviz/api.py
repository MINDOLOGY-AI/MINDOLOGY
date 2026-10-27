# api.py -- the probes tab from python: no server, browser or graph trace (see interpviz/DOC.md, api).
# run from repo root, e.g.
#   iv = Interp("interpviz/models/qwen3_5_4b", "wcc_resid_jumprelu")
#   r = iv.run("The Eiffel Tower is in the city of")              # = the sae_run reply's "run" + "reply"
#   a = iv.probes.attribute(iv.model, len(r["tokens"]) - 1)        # sae_attribute
#   c = iv.probes.feature(iv.model, iv.set, "L12", 4242)           # sae_feature (sae_search, sae_neuron likewise)
#   e = [{"kind": "feature", "set": iv.set, "part": "L12", "pos": 9, "id": 4242, "value": 0.0}]   # edits: interpviz/DOC.md, session
#   iv.next_logprobs("The Eiffel Tower is in the city of", [" Paris", " Rome"], edits=e)

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

    def run(self, text, edits=(), n_generate=0, template=False, think=False):
        # one user message (template: in qwen's chat template, else plain text) + up to n_generate reply tokens.
        # edits: [edit] wire format (interpviz/DOC.md, session) -> {**the sae_run reply's "run", "reply": str}
        s = self.probes.run(self.model, self.set, [{"role": "user", "content": text}], not template, think, n_generate, False, list(edits))
        return {**s["run"], "reply": s["query"]["messages"][-1]["content"] if n_generate else ""}

    def next_logprobs(self, text, words, edits=()):
        # log prob of each word's first token as the next token after text (plain, no template), with edits applied
        # -> {word: float}. needs the set loaded (one run first)
        assert self.probes.hook_set is not None, "run a prompt first (loads the hook set)"
        tok = self.probes.tokenizer
        device = next(self.model.parameters()).device
        # (1, T)
        ids = torch.tensor([([tok.bos_token_id] if tok.bos_token_id is not None else []) + tok.encode(text)], device=device)
        with torch.no_grad(), edited(self.model, self.probes.saes, self.probes.resolve(list(edits))):
            # (1, T, V) -> (V,)
            lp = self.model(ids)[0, -1].float().log_softmax(-1)
        return {w: lp[tok.encode(w)[0]].item() for w in words}
