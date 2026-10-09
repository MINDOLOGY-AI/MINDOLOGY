# attribution of SAE features to one next-token prediction (see synapse/interp/DOC.md, attribution).
# one forward + one backward on the real model (nothing spliced): g = d target / d x at every SAE's read point, then
# every active feature gets a_i · (d_i · g), d_i = its decoder row / s (only the k active rows are gathered), and every
# (point, token) a bias node (b_dec / s) · g and an error node (x - x̂) · g. features + bias + error = x · g exactly.
# scores are total effects (every downstream path), first order.
# model-agnostic: a causal LM model(ids) -> logits (1, T, V) and TopK SAEs (encode(x) -> vals, idx; W_dec; norm_factor).

import torch


def attribute(model, ids, saes, target_pos):
    # ids: (1, T) token ids on the model's device
    # saes: {name: (point, sae)}; point = "<module path>:<in|out>" of the activation the sae reads, e.g. "model.layers.8:out"
    #       (a module returning a tuple, like a decoder layer, has the activation as its first element)
    # target_pos: token column t to explain: target = the logit of ids[0, t] at position t - 1 (the prediction of it)
    # -> {"logit": float, "parts": {name: {"vals": (T, k), "idx": (T, k), "attr": (T, k), "bias": (T,), "err": (T,)}}}, float32 / long cpu
    T = ids.shape[1]
    assert 1 <= target_pos < T, f"target_pos {target_pos} must be in [1, {T})"
    # {point: (1, T, d) the activation tensor the model passes on (a slice of it would not be in the graph)}
    acts = {}
    handles = []  # [RemovableHandle]
    for point in {p for p, _ in saes.values()}:
        path, side = point.rsplit(":", 1)
        module = model.get_submodule(path)
        if side == "out":
            handles.append(module.register_forward_hook(
                lambda mod, args, out, point=point: acts.__setitem__(point, out[0] if isinstance(out, tuple) else out)))
        else:
            assert side == "in", f"bad point {point!r}: must end in :in or :out"
            handles.append(module.register_forward_pre_hook(lambda mod, args, point=point: acts.__setitem__(point, args[0])))
    try:
        with torch.enable_grad():
            # (1, T, V)
            logits = model(ids)
            target = logits[0, target_pos - 1, ids[0, target_pos]]
            assert target.requires_grad, "model parameters must require grad (autograd needs a graph to differentiate)"
            points = sorted(acts)
            # {point: (1, T, d) d target / d activation}; weight gradients are never computed
            grads = dict(zip(points, torch.autograd.grad(target, [acts[p] for p in points])))
    finally:
        for h in handles:
            h.remove()

    parts = {}  # {name: {"vals", "idx", "attr", "bias", "err"}}
    with torch.no_grad():
        for name, (point, sae) in saes.items():
            dev = sae.W_dec.device
            # (1, T, d) -> (T, d) activation and its gradient, fp32 on the sae's device
            x = acts[point][0].detach().float().to(dev)
            g = grads[point][0].float().to(dev)
            # (T, d) -> (T, k), (T, k)
            vals, idx = sae.encode(x)
            # (T, k, d) the active features' decoder rows in model scale
            rows = sae.W_dec[idx] / sae.norm_factor
            # (T, k, d) · (T, d) -> (T, k)
            attr = vals * torch.einsum("tkd,td->tk", rows, g)
            # (d,) the sae's constant, in model scale
            b = sae.b_dec / sae.norm_factor
            # (T, d) -> (T,)
            bias = (b * g).sum(-1)
            # (T, d) what the sae leaves unexplained, x - x̂ -> (T,)
            err = ((x - torch.einsum("tk,tkd->td", vals, rows) - b) * g).sum(-1)
            parts[name] = {"vals": vals.cpu(), "idx": idx.cpu(), "attr": attr.cpu(), "bias": bias.cpu(), "err": err.cpu()}
    return {"logit": target.item(), "parts": parts}
