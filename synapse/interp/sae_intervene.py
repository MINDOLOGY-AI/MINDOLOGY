# intervention: set SAE features to chosen values inside a model's forward (see synapse/interp/DOC.md, intervention).
# while the context is active every forward of the model (a single run, each generation step, an attribution) sets
# feature f of an sae at token pos to a value by adding (value - current) · W_dec[f] / s to the activation the sae reads;
# current = the feature's value among the sae's top-k on that (possibly already edited upstream) activation, 0 if it is
# not active. the rest of the activation, the sae's error included, is untouched.
# model-agnostic: any torch model with submodules, TopK SAEs (encode(x) -> vals, idx; W_dec; norm_factor).
#   with edited(model, saes, [("L15", 5, 9443, 0.0)]):
#       logits = model(ids)

import contextlib

import torch


@contextlib.contextmanager
def edited(model, saes, edits):
    # saes: {name: (point, sae)}; point = "<module path>:<in|out>" of the activation the sae reads, e.g. "model.layers.8:out"
    # edits: [(name, pos, feature id, value)]; applied in list order, so two edits on one point see each other
    # {point: [(sae, pos, feature id, value)]}
    by_point = {}
    for name, pos, fid, value in edits:
        point, sae = saes[name]
        by_point.setdefault(point, []).append((sae, pos, fid, value))

    def apply(h, es):
        # h: (1, T, d) the activation at a point -> h + the edits' deltas (a constant for autograd)
        # (1, T, d) sum of the deltas, fp32 until it is added
        delta = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
        with torch.no_grad():
            for sae, pos, fid, value in es:
                # a generated token's edit starts once generation has reached that position
                if pos >= h.shape[1]:
                    continue
                # (1, d) the token's activation with the edits so far, on the sae's device
                x = (h[0, pos].float() + delta[0, pos]).to(sae.W_dec.device)[None]
                vals, idx = sae.encode(x)
                current = vals[0][idx[0] == fid].sum().item()
                delta[0, pos] += ((value - current) * sae.W_dec[fid] / sae.norm_factor).to(h.device)
        return h + delta.to(h.dtype)

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
