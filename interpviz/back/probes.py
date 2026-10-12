# probes.py -- the probes tab's requests without the websocket (see interpviz/DOC.md, probes tab).
# server.py forwards its sae_* messages here; interpviz/api.py drives it directly from python (no fastapi needed).
# the caller owns the model (and its device / dtype) and passes it to every request that runs it.

import importlib
from pathlib import Path

import torch
from transformers.convert_slow_tokenizer import bytes_to_unicode

from interpviz.back.hooks import HookSet
from synapse.interp.sae_attribute import attribute
from synapse.interp.sae_intervene import edited

# top next-token guesses shown per position
N_NEXT = 5
# tokens a latent card lists as most pushed toward / away from
N_LOGITS = 10
# most results a label search returns
SEARCH_LIMIT = 50


class Probes:
    def __init__(self, meta, folder):
        # meta: the model folder's interpvizMeta.json (dict), folder: its path (hook set files are relative to it)
        self.meta = meta
        self.folder = Path(folder)
        # "module.path.function" that returns the tokenizer
        mod_path, fn = meta["model"]["tokenizer"].rsplit(".", 1)
        self.tokenizer = getattr(importlib.import_module(mod_path), fn)()
        # {byte-level BPE char: byte}, inverse of the tokenizer's byte -> printable-char map
        byte_of = {c: b for b, c in bytes_to_unicode().items()}
        # [bytes] raw utf-8 bytes per token id
        self.token_bytes = [bytes(byte_of[c] for c in self.tokenizer.convert_ids_to_tokens(i)) for i in range(len(self.tokenizer))]
        # {set name: hook set file relative to the folder}
        self.files = {Path(f).stem: f for f in meta.get("hooks", [])}
        # the one loaded HookSet: the set the last run asked for (probe weights are GBs), replaced when another is asked
        self.hook_set = None
        # {(set, part): (read point, probe)} every part of the loaded set, the saes argument of attribute / edited
        self.saes = None
        # (1, T) token ids of the last run (prompt + generated), what attribute explains
        self.ids = None
        # [((set, part), pos, feature id, value)] the feature edits of the last run, also applied by attribute
        self.edits = []

    def run(self, model, text, template, n_generate, edits, set_name):
        # text -> tokens (+ n_generate greedy tokens) -> one forward capturing every read point of hook set set_name
        # edits: [[set, part, pos, feature id, value]] (see synapse/interp/sae_intervene.py)
        # -> {tokens, n_prompt, output, next, hooks} (interpviz/DOC.md, sae_run)
        assert set_name in self.files, f"no hook set {set_name!r} in interpvizMeta.json hooks, have {list(self.files)}"
        if self.hook_set is None or self.hook_set.name != set_name:
            # drop the old set's GBs before loading the next
            self.hook_set = None
            self.saes = None
            self.hook_set = HookSet(self.folder / self.files[set_name])
            self.saes = {(self.hook_set.name, pn): (part["reads"], part["probe"]) for pn, part in self.hook_set.parts.items()}
        self.edits = [((sn, pn), pos, fid, value) for sn, pn, pos, fid, value in edits]
        tok = self.tokenizer
        device = next(model.parameters()).device
        if template:
            # the chat template starts with bos itself
            prompt = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                             tokenize=True, return_dict=True)["input_ids"]
        else:
            # models without a bos token (qwen) start straight with the text
            prompt = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + tok.encode(text)
        # (1, T) token ids, grown by greedy generation
        ids = torch.tensor([prompt], device=device)
        special_ids = set(tok.all_special_ids)  # {int}
        with torch.no_grad(), edited(model, self.saes, self.edits):
            for _ in range(n_generate):
                # (1, T, V) -> (1, 1)
                nxt = model(ids)[:, -1].argmax(-1, keepdim=True)
                ids = torch.cat([ids, nxt], dim=1)
                # any special token ends it (qwen: <|im_end|> after a chat turn, <|endoftext|> after plain text)
                if nxt.item() in special_ids:
                    break

        # {read point: (T, d) float32 cpu} filled by forward hooks on the read points' modules. the edit hooks go in
        # first, so the capture hooks (run after them) see the edited activations
        captured = {}
        handles = []  # [RemovableHandle]
        with torch.no_grad(), edited(model, self.saes, self.edits):
            for point in {part["reads"] for part in self.hook_set.parts.values()}:
                path, side = point.rsplit(":", 1)
                module = model.get_submodule(path)
                if side == "out":
                    # decoder layers return (hidden_states, present_kv): the activation is the first element
                    handles.append(module.register_forward_hook(
                        lambda mod, args, out, point=point: captured.__setitem__(point, (out[0] if isinstance(out, tuple) else out)[0].float().cpu())))
                else:
                    handles.append(module.register_forward_pre_hook(
                        lambda mod, args, point=point: captured.__setitem__(point, args[0][0].float().cpu())))
            try:
                # (1, T, V)
                logits = model(ids)
            finally:
                for h in handles:
                    h.remove()

        self.ids = ids
        # (T, V) -> (T, N_NEXT)
        probs, nxt = logits[0].float().softmax(-1).topk(N_NEXT, dim=-1)
        return {
            "tokens": [tok.decode([i]) for i in ids[0].tolist()],
            "n_prompt": len(prompt),
            # the generated continuation as text
            "output": tok.decode(ids[0, len(prompt):].tolist()),
            "next": [[[tok.decode([i]), round(p, 3)] for i, p in zip(ir, pr)] for ir, pr in zip(nxt.tolist(), probs.tolist())],
            "hooks": {self.hook_set.name: {"display": self.hook_set.display, "editable": self.hook_set.editable,
                                           "parts": self.hook_set.encode(captured)}},
        }

    def attribute(self, model, target):
        # attribution of every part's active latents (+ bias and error node per cell) to the prediction of token column
        # target of the last run, with its edits (see synapse/interp/sae_attribute.py)
        # -> {target, logit, parts: {set: {part: {ids, attr, bias, err}}}}
        assert self.ids is not None, "run a prompt first"
        with edited(model, self.saes, self.edits):
            r = attribute(model, self.ids, self.saes, target)
        parts = {}  # {set: {part: {"ids": [[int]], "attr": [[float]], "err": [float], "bias": [float]}}}, per token strongest activation first, zeros dropped
        for (sn, pn), p in r["parts"].items():
            n = self.hook_set.display["max_features"]
            keep = p["vals"][:, :n] > 0
            parts.setdefault(sn, {})[pn] = {
                "ids": [row[k].tolist() for row, k in zip(p["idx"][:, :n], keep)],
                "attr": [[round(a, 4) for a in row[k].tolist()] for row, k in zip(p["attr"][:, :n], keep)],
                "err": [round(e, 4) for e in p["err"].tolist()],
                "bias": [round(b, 4) for b in p["bias"].tolist()],
            }
        return {"target": target, "logit": round(r["logit"], 3), "parts": parts}

    def search(self, set_name, part, query):
        # labels of one hook part containing every word of query, best label score first
        assert self.hook_set is not None and self.hook_set.name == set_name, "run a prompt with this hook set first"
        return {"results": self.hook_set.search(part, query, SEARCH_LIMIT)}

    def feature(self, model, set_name, part, unit):
        # one latent's card: label, score, density, example windows from its picks, and (if the meta names the model's
        # final norm and unembedding in model.logit_lens) the tokens its write pushes the next-token logits toward / away from
        assert self.hook_set is not None and self.hook_set.name == set_name, "run a prompt with this hook set first"
        card = self.hook_set.feature(part, unit, self.token_bytes)
        lens = self.meta["model"].get("logit_lens")
        if lens:
            norm, head = model.get_submodule(lens["norm"]), model.get_submodule(lens["unembed"])
            # (d_in,) the latent's write times the final norm's per-dim gain (lens["centered"]: the norm scales by 1 + weight,
            # qwen3.5). the norm's per-token 1 / rms is left out, so the numbers rank tokens, they are not exact logit changes
            v = self.hook_set.out_dir(part, unit) * (norm.weight.float().cpu() + (1 if lens["centered"] else 0))
            # (V, d_in) @ (d_in,) -> (V,)
            eff = (head.weight @ v.to(head.weight)).float().cpu()
            up, down = eff.topk(N_LOGITS), (-eff).topk(N_LOGITS)
            card["logits"] = {"up": [[self.tokenizer.decode([i]), round(x, 3)] for x, i in zip(up.values.tolist(), up.indices.tolist())],
                              "down": [[self.tokenizer.decode([i]), round(-x, 3)] for x, i in zip(down.values.tolist(), down.indices.tolist())]}
        return card
