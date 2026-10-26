# intervention: edit activations inside a model's forward (see synapse/interp/DOC.md, intervention).
# while the context is active every forward of the model (a single run, each generation step, an attribution) applies
# the edits at their points, at one token or at every token. three kinds:
#   feature: set feature f of an sae to a value by adding (value - current) · W_dec[f] / s to the activation the sae
#            reads; current = the feature's value among the sae's top-k on that (possibly already edited upstream)
#            activation, 0 if it is not active. the rest of the activation, the sae's error included, is untouched
#   neuron:  set unit u of the activation at a point to a value (e.g. an mlp neuron: the input of its down projection)
#   add:     add a vector to the activation at a point (steering, e.g. a feature's write times a strength)
# model-agnostic: any torch model with submodules, TopK SAEs / crosscoder layers (encode(x) -> vals, idx; W_dec; norm_factor).
#   with edited(model, saes, [{"kind": "feature", "sae": "L15", "pos": 5, "id": 9443, "value": 0.0}]):
#       logits = model(ids)

import contextlib

import torch


@contextlib.contextmanager
def edited(model, saes, edits):
    # saes: {name: (point, sae)}; point = "<module path>:<in|out>" of the activation the sae reads, e.g. "model.layers.8:out"
    # edits: [{"kind": "feature", "sae": name, "pos": int | None, "id": int, "value": float}
    #       | {"kind": "neuron", "point": str, "pos": int | None, "unit": int, "value": float}
    #       | {"kind": "add", "point": str, "pos": int | None, "vec": (d,) tensor}]
    # pos None = every token (generated ones included). applied in list order, so two edits on one point see each other
    # {point: [edit]}
    by_point = {}
    for e in edits:
        assert e["kind"] in ("feature", "neuron", "add"), f"unknown edit kind {e['kind']!r}"
        point = saes[e["sae"]][0] if e["kind"] == "feature" else e["point"]
        by_point.setdefault(point, []).append(e)

    def apply(h, es):
        # h: (1, T, d) the activation at a point -> h with the edits (the edits are constants for autograd)
        T = h.shape[1]
        # (1, T, d) sum of the additive deltas, fp32 until it is added
        delta = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
        # [(unit, [token], value)] neuron sets, applied after the deltas
        sets = []
        with torch.no_grad():
            for e in es:
                # a generated token's edit starts once generation has reached that position
                if e["pos"] is not None and e["pos"] >= T:
                    continue
                ts = list(range(T)) if e["pos"] is None else [e["pos"]]
                if e["kind"] == "neuron":
                    sets.append((e["unit"], ts, e["value"]))
                elif e["kind"] == "add":
                    delta[0, ts] += e["vec"].to(h.device, torch.float32)
                else:
                    sae = saes[e["sae"]][1]
                    # (n, d) the tokens' activations with the edits so far, on the sae's device
                    x = (h[0, ts].float() + delta[0, ts]).to(sae.W_dec.device)
                    # (n, k), (n, k)
                    vals, idx = sae.encode(x)
                    # (n,) the feature's current value per token, 0 where it is not active
                    current = (vals * (idx == e["id"])).sum(-1)
                    # (n, 1) * (d,) -> (n, d)
                    delta[0, ts] += ((e["value"] - current)[:, None] * sae.W_dec[e["id"]].float() / sae.norm_factor).to(h.device)
        h = h + delta.to(h.dtype)
        if sets:
            # a set unit is a constant: no gradient flows through it
            h = h.clone()
            for unit, ts, value in sets:
                h[0, ts, unit] = value
        return h

    handles = []  # [RemovableHandle]
    for point, es in by_point.items():
        path, side = point.rsplit(":", 1)
        module = model.get_submodule(path)
        if side == "out":
            # a module returning a tuple (a decoder layer) has the activation as its first element
            handles.append(module.register_forward_hook(
                lambda mod, args, out, es=es: (apply(out[0], es), *out[1:]) if isinstance(out, tuple) else apply(out, es)))
        else:
            assert side == "in", f"bad point {point!r}: must end in :in or :out"
            handles.append(module.register_forward_pre_hook(lambda mod, args, es=es: (apply(args[0], es), *args[1:])))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
