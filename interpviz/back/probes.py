# probes.py -- the session every frontend shares (prompt, edits, last run) and the probes tab's requests, without the
# websocket (see interpviz/DOC.md, session / probes tab).
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
# tokens a latent / neuron card lists as most pushed toward / away from
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
        # {"point": "<module path with {layer}>:in", "layers": int, "units": int} where the model's neurons are (the
        # input of a linear layer, whose weight column u is neuron u's write), or None: no neuron edits
        self.neurons = meta["model"].get("neurons")
        if self.neurons:
            assert self.neurons["point"].endswith(":in"), "neurons point must be the input of a linear layer (:in)"
        # {set name: hook set file relative to the folder}
        self.files = {Path(f).stem: f for f in meta.get("hooks", [])}
        # the one loaded HookSet: the set the last run asked for (probe weights are GBs), replaced when another is asked
        self.hook_set = None
        # {(set, part): (read point, probe)} every part of the loaded set, the saes argument of attribute / edited
        self.saes = {}
        # (1, T) token ids of the last run (prompt + generated), what attribute and neuron explain
        self.ids = None
        # the session (interpviz/DOC.md, session): the last run's request {"set": str | None, "messages" (with the reply),
        # "raw", "think", "max_new"}, its edits [edit] (wire format) and its reply {tokens, n_prompt, next, hooks}
        self.query = None
        self.edits = []
        self.last = None

    def session(self):
        # -> {"query", "edits", "run"} what every frontend shows, or None before the first run
        return {"query": self.query, "edits": self.edits, "run": self.last} if self.last else None

    def resolve(self, edits):
        # wire edits -> synapse/interp/sae_intervene.py edits. pos "all" -> None (every token)
        out = []  # [{"kind": "feature" | "neuron" | "add", ...}]
        for e in edits:
            pos = None if e["pos"] == "all" else e["pos"]
            if e["kind"] in ("neuron", "neuron_add"):
                assert self.neurons, "this model's meta names no neurons (model.neurons)"
                assert 0 <= e["layer"] < self.neurons["layers"] and 0 <= e["unit"] < self.neurons["units"], f"no neuron {e['layer']}:{e['unit']}"
                point = self.neurons["point"].format(layer=e["layer"])
                if e["kind"] == "neuron":
                    out.append({"kind": "neuron", "point": point, "pos": pos, "unit": e["unit"], "value": e["value"]})
                else:
                    # (units,) value at the neuron, 0 elsewhere: adds value × its weight column to the layer's output
                    vec = torch.zeros(self.neurons["units"])
                    vec[e["unit"]] = e["value"]
                    out.append({"kind": "add", "point": point, "pos": pos, "vec": vec})
                continue
            assert self.hook_set is not None and e["set"] == self.hook_set.name, \
                f"edit on hook set {e['set']!r}, but {self.hook_set.name if self.hook_set else None!r} is loaded"
            if e["kind"] == "feature":
                out.append({"kind": "feature", "sae": (e["set"], e["part"]), "pos": pos, "id": e["id"], "value": e["value"]})
            else:
                assert e["kind"] == "steer", f"unknown edit kind {e['kind']!r}"
                # a crosscoder feature adds its write into every layer it writes to
                out += [{"kind": "add", "point": point, "pos": pos, "vec": e["value"] * v} for point, v in self.hook_set.writes(e["part"], e["id"])]
        return out

    def run(self, model, set_name, messages, raw, think, max_new, edits, on_reply=None, stop=None):
        # a conversation -> tokens (+ the model's greedy reply, up to max_new tokens, when the last message is the
        # user's) -> one forward capturing every read point of hook set set_name (None: no hook set, tokens and
        # next-token guesses only). edits: [edit] wire format (interpviz/DOC.md, session)
        # messages: [{"role": "user" | "assistant", "content": str}]; raw: the contents joined as plain text (the reply
        # continues it), else qwen's chat template, think: its thinking block (off: an empty one)
        # on_reply(str): called with the reply so far after every generated token (streaming); stop: a threading.Event,
        # once set the reply ends where it is (it is kept, and read like a finished one)
        # -> the new session {query, edits, run}; query.messages ends with the reply
        assert messages and all(m["role"] in ("user", "assistant") for m in messages), "messages: [{role: user | assistant, content}]"
        assert set_name is None or set_name in self.files, f"no hook set {set_name!r} in interpvizMeta.json hooks, have {list(self.files)}"
        if set_name is not None and (self.hook_set is None or self.hook_set.name != set_name):
            # drop the old set's GBs before loading the next
            self.hook_set = None
            self.saes = {}
            self.hook_set = HookSet(self.folder / self.files[set_name])
            self.saes = {(self.hook_set.name, pn): (part["reads"], part["probe"]) for pn, part in self.hook_set.parts.items()}
        tok = self.tokenizer
        device = next(model.parameters()).device
        reply = messages[-1]["role"] == "user" and max_new > 0
        if raw:
            # models without a bos token (qwen) start straight with the text
            prompt = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + tok.encode("".join(m["content"] for m in messages))
        else:
            # the chat template starts with bos itself; enable_thinking=False writes an empty <think></think>
            prompt = tok.apply_chat_template(messages, add_generation_prompt=messages[-1]["role"] == "user", enable_thinking=think,
                                             tokenize=True, return_dict=True)["input_ids"]
        # a token edit survives where its token is the last run's (the tokens before it unchanged too), or is still to
        # be generated after an unchanged prompt (a regenerated reply). e.g. a new turn: qwen's template drops the
        # empty think block of earlier replies, so their tokens move and their edits go
        old = self.ids[0].tolist() if self.ids is not None else []  # [int]
        # length of the prefix the new prompt shares with the last run's tokens
        common = next((i for i, (a, b) in enumerate(zip(old, prompt)) if a != b), min(len(old), len(prompt)))
        edits = [e for e in edits if e["pos"] == "all" or e["pos"] < common or (common == len(prompt) and e["pos"] >= len(prompt))]
        resolved = self.resolve(edits)
        # (1, T) token ids, grown by greedy generation
        ids = torch.tensor([prompt], device=device)
        special_ids = set(tok.all_special_ids)  # {int}
        with torch.no_grad(), edited(model, self.saes, resolved):
            for _ in range(max_new if reply else 0):
                if stop is not None and stop.is_set():
                    break
                # (1, T, V) -> (1, 1)
                nxt = model(ids)[:, -1].argmax(-1, keepdim=True)
                ids = torch.cat([ids, nxt], dim=1)
                # any special token ends it (qwen: <|im_end|> after a chat turn, <|endoftext|> after plain text)
                if nxt.item() in special_ids:
                    break
                if on_reply:
                    on_reply(tok.decode(ids[0, len(prompt):].tolist()))
        if reply:
            # the reply without its closing special token
            gen = [i for i in ids[0, len(prompt):].tolist() if i not in special_ids]
            messages = [*messages, {"role": "assistant", "content": tok.decode(gen)}]

        # {read point: (T, d) float32 cpu} filled by forward hooks on the read points' modules. the edit hooks go in
        # first, so the capture hooks (run after them) see the edited activations
        captured = {}
        handles = []  # [RemovableHandle]
        points = {part["reads"] for part in self.hook_set.parts.values()} if set_name is not None else set()  # {str}
        with torch.no_grad(), edited(model, self.saes, resolved):
            for point in points:
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
        self.query = {"set": set_name, "messages": messages, "raw": raw, "think": think, "max_new": max_new}
        self.edits = edits
        self.last = {
            "tokens": [tok.decode([i]) for i in ids[0].tolist()],
            "n_prompt": len(prompt),
            "next": [[[tok.decode([i]), round(p, 3)] for i, p in zip(ir, pr)] for ir, pr in zip(nxt.tolist(), probs.tolist())],
            "hooks": {} if set_name is None else {self.hook_set.name: {
                "display": self.hook_set.display, "editable": self.hook_set.editable, "parts": self.hook_set.encode(captured)}},
        }
        return self.session()

    def attribute(self, model, target):
        # attribution of every part's active latents (+ bias and error node per cell) to the prediction of token column
        # target of the last run, with its edits (see synapse/interp/sae_attribute.py)
        # -> {target, logit, parts: {set: {part: {ids, attr, bias, err}}}}
        assert self.ids is not None, "run a prompt first"
        assert self.query["set"] is not None, "the last run used no hook set"
        with edited(model, self.saes, self.resolve(self.edits)):
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

    def logit_lens(self, model, v):
        # v (d_in,) fp32 cpu: a model-scale write into the residual stream -> {"up": [[token, float]], "down": [...]}: the
        # N_LOGITS tokens it pushes the next-token logits toward / away from, or None if the meta names no
        # model.logit_lens (the model's final norm and unembedding)
        lens = self.meta["model"].get("logit_lens")
        if not lens:
            return None
        norm, head = model.get_submodule(lens["norm"]), model.get_submodule(lens["unembed"])
        # (d_in,) the write times the final norm's per-dim gain (lens["centered"]: the norm scales by 1 + weight, qwen3.5).
        # the norm's per-token 1 / rms is left out, so the numbers rank tokens, they are not exact logit changes
        v = v * (norm.weight.float().cpu() + (1 if lens["centered"] else 0))
        # (V, d_in) @ (d_in,) -> (V,)
        eff = (head.weight @ v.to(head.weight)).float().cpu()
        up, down = eff.topk(N_LOGITS), (-eff).topk(N_LOGITS)
        return {"up": [[self.tokenizer.decode([i]), round(x, 3)] for x, i in zip(up.values.tolist(), up.indices.tolist())],
                "down": [[self.tokenizer.decode([i]), round(-x, 3)] for x, i in zip(down.values.tolist(), down.indices.tolist())]}

    def feature(self, model, set_name, part, unit):
        # one latent's card: label, score, density, example windows from its picks, and the tokens its write (the one
        # reaching the logits most directly, hooks.py writes) pushes the next-token logits toward / away from
        assert self.hook_set is not None and self.hook_set.name == set_name, "run a prompt with this hook set first"
        card = self.hook_set.feature(part, unit, self.token_bytes)
        logits = self.logit_lens(model, self.hook_set.writes(part, unit)[-1][1])
        if logits:
            card["logits"] = logits
        return card

    def neuron(self, model, layer, unit):
        # one neuron's card: its value at every token of the last run (with its edits) and the tokens its write (the
        # linear layer's weight column) pushes the next-token logits toward / away from
        # -> {"layer", "unit", "acts": [float] per token, "logits": {up, down} | None}
        assert self.ids is not None, "run a prompt first"
        point = self.resolve([{"kind": "neuron", "layer": layer, "unit": unit, "pos": "all", "value": 0.0}])[0]["point"]
        module = model.get_submodule(point.rsplit(":", 1)[0])
        # [(T,) float32 cpu] the neuron's input value per token
        acts = []
        with torch.no_grad(), edited(model, self.saes, self.resolve(self.edits)):
            # registered after the edit hooks, so it sees the edited value
            h = module.register_forward_pre_hook(lambda mod, args: acts.append(args[0][0, :, unit].float().cpu()))
            try:
                model(self.ids)
            finally:
                h.remove()
        # (d_out, d_in) weight -> (d_out,) column: the neuron's write
        v = module.weight[:, unit].detach().float().cpu()
        return {"layer": layer, "unit": unit, "acts": [round(a, 3) for a in acts[0].tolist()], "logits": self.logit_lens(model, v)}
