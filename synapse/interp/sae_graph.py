# attribution graph of SAE / crosscoder features for one next-token prediction (writing/interp/graph_pruning.md).
# nodes: every active feature at every (layer, token) up to the predicting position, per (layer, token) an error node
# (x - x̂) and a bias node (the dictionary's constant), per token an embedding node, and the target logit.
# edges are direct, through one decoder layer with the routing frozen (attention q / k and query gate, every rmsnorm's
# 1/rms, gated deltanet q / k / β / decay / output gate): A[t, s] = (s's write into the residual t reads) · ∂pre_t/∂x.
# the MLPs stay live, so an edge is first order (exact only if the layer were linear).
# influence on the logit over every path, B = A + A² + ... = (I - A)^-1 - I on |A| with each node's incoming edges
# normalized to sum to 1, as one top-down pass (the graph is a DAG): infl(s) = Â[logit, s] + Σ_t infl(t) Â[t, s].
# every unit of influence ends in a root (error / bias / embedding: nothing points into them), so roots sum to 1.
# dictionaries: saes[m] reads layers[m]'s output. a TopK SAE writes only into its own layer. a weakly causal crosscoder
# layer (WCCEncoder) writes into its own and every later layer, so the residual at m splits into every crosscoder
# feature of layers <= m (one node per feature, no copy per layer).
# the freezing hooks know synapse's RMSNorm / GQA / GatedDeltaNet modules (Qwen3.5); other models need their own.

import torch

from synapse.algorithms.attention.GQA import GQA
from synapse.algorithms.attention.GatedDeltaNet import GatedDeltaNet
from synapse.algorithms.basic.RMSnorm import RMSNorm


def freeze_hooks(model):
    # hooks that keep every forward value but stop gradients through the routing -> [RemovableHandle]
    def detach(mod, args, out):
        return out.detach()

    def norm(mod, args, out):
        x = args[0].float()
        # (..., d) -> (..., 1) frozen 1/rms
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + mod.eps).detach()
        return (x * r * (1.0 + mod.weight) if mod.centered else mod.weight * (x * r)).to(args[0].dtype)

    def qk_channels(mod, args, out):
        # gated deltanet conv output (batch, conv_dim, seq) = [q | k | v] channels before the silu: q, k frozen, v live
        n = mod.n_qk
        return torch.cat([out[:, :n].detach(), out[:, n:]], 1)

    handles = []  # [RemovableHandle]
    for mod in model.modules():
        if isinstance(mod, RMSNorm):
            assert mod.cast_to_float32, "freeze hook mirrors the fp32 rmsnorm path only"
            handles.append(mod.register_forward_hook(norm))
        elif isinstance(mod, GQA):
            # q_proj holds the query and the query gate
            handles += [mod.q_proj.register_forward_hook(detach), mod.k_proj.register_forward_hook(detach)]
        elif isinstance(mod, GatedDeltaNet):
            mod.conv1d.n_qk = 2 * mod.key_dim
            handles.append(mod.conv1d.register_forward_hook(qk_channels))
            # z = output gate, b -> β, a -> decay
            handles += [mod.in_proj_z.register_forward_hook(detach), mod.in_proj_b.register_forward_hook(detach),
                        mod.in_proj_a.register_forward_hook(detach)]
    return handles


def rep(v, N):
    # a layer call's argument with batch 1 -> batch N (tensors expanded, tuples recursed, the rest as is)
    if torch.is_tensor(v):
        return v.expand(N, *v.shape[1:]) if v.shape[0] == 1 else v
    return tuple(rep(u, N) for u in v) if isinstance(v, tuple) else v


def build_graph(model, layers, saes, ids, target_pos, chunk=64, top_edges=64, node_frac=0.8, edge_frac=0.98):
    # model(ids) -> logits (1, T, V); layers: its decoder layers in order, layers[m](hidden, ...) -> hidden (1, T, D)
    # saes: saes[m] reads layers[m]'s output; TopKSAE (encode, W_enc (D, F), W_dec (F, D), b_dec, norm_factor) or
    #       WCCEncoder (acts, W_enc, W_dec_all (F, L - m, D), out_scale, b_dec, norm_factor)
    # target_pos: explain the logit of ids[0, target_pos] at position target_pos - 1
    # chunk: targets per batched backward; top_edges: incoming edges kept per node for the edge pruning
    # node_frac / edge_frac: keep the most influential nodes / edges covering this fraction of the total
    # -> {"logit", "prob", "roots": {"error", "bias", "embed"} influence shares, "n_nodes", "nodes": [kept node dicts],
    #     "edges": [{"src", "dst", "weight" (raw signed A), "score"}]}
    L, T = len(layers), ids.shape[1]
    assert len(saes) == L, f"{len(saes)} dictionaries for {L} layers"
    assert 1 <= target_pos < T, f"target_pos {target_pos} must be in [1, {T})"
    P = target_pos - 1  # the predicting position; later tokens can't affect the target
    wcc = hasattr(saes[0], "W_dec_all")
    assert all(hasattr(s, "W_dec_all") == wcc for s in saes), "mixed dictionary kinds"
    dev = ids.device

    # ---- one frozen forward: every layer's output and call ----
    xs = [None] * L  # [(1, T, D)] layer outputs, in the autograd graph
    calls = [None] * L  # [(args, kwargs)] each layer's call without the hidden state
    emb = []  # [(1, T, D)] layer 0's input, a leaf so everything after it is differentiable

    def pre(mod, args, kwargs):
        calls[mod.graph_idx] = (args[1:], kwargs)
        if mod.graph_idx == 0:
            emb.append(args[0].detach().requires_grad_())
            return (emb[0], *args[1:]), kwargs

    def post(mod, args, out):
        assert torch.is_tensor(out), "a layer must return the hidden state tensor"
        xs[mod.graph_idx] = out

    handles = freeze_hooks(model)
    try:
        capture = []  # [RemovableHandle] only for the main forward (the per-layer reruns must not overwrite xs)
        for m, layer in enumerate(layers):
            layer.graph_idx = m
            capture += [layer.register_forward_pre_hook(pre, with_kwargs=True), layer.register_forward_hook(post)]
        with torch.enable_grad():
            try:
                # (1, T, V)
                logits = model(ids)
            finally:
                for hd in capture:
                    hd.remove()
            logit = logits[0, P, ids[0, target_pos]]
            prob = logits[0, P].float().softmax(-1)[ids[0, target_pos]].item()
            # (T, D) ∂logit/∂x at the last layer (final norm frozen, unembedding)
            g_logit = torch.autograd.grad(logit, xs[L - 1])[0][0].float()

        # ---- feature nodes: every active feature at positions <= P ----
        nodes = []  # [{"kind", "layer", "pos", "feature", "act"}] features first, roots appended later
        feats = []  # per encoder layer: {"pos": (n,) long, "f": (n,) long, "a": (n,) fp32, "node": (n,) long}, model device
        with torch.no_grad():
            for i, sae in enumerate(saes):
                # (P+1, D) model-scale residual the dictionary reads
                x = xs[i][0, :P + 1].float().to(sae.W_enc.device)
                if wcc:
                    # (P+1, F) every firing jumprelu feature (encode would cap at k)
                    a = sae.acts(x)
                    pos, f = a.nonzero(as_tuple=True)
                    a = a[pos, f]
                else:
                    # (P+1, k), (P+1, k)
                    vals, idx = sae.encode(x)
                    keep = vals > 0
                    pos = torch.arange(P + 1, device=x.device)[:, None].expand_as(idx)[keep]
                    f, a = idx[keep], vals[keep]
                node = torch.arange(len(nodes), len(nodes) + len(f))
                nodes += [{"kind": "feature", "layer": i, "pos": p, "feature": u, "act": v}
                          for p, u, v in zip(pos.tolist(), f.tolist(), a.tolist())]
                feats.append({"pos": pos.to(dev), "f": f.to(dev), "a": a.to(dev), "node": node.to(dev)})
        n_feat = len(nodes)
        # root nodes: error / bias of residual m at position q, embedding at q
        root = {}  # {("error" | "bias", m, q) or ("embed", -1, q): node index}
        for kind, layer_range in (("error", range(L)), ("bias", range(L)), ("embed", [-1])):
            for m in layer_range:
                for q in range(P + 1):
                    root[(kind, m, q)] = len(nodes)
                    nodes.append({"kind": kind, "layer": m, "pos": q, "feature": None, "act": None})
        infl = torch.zeros(len(nodes), device=dev)  # (n_nodes,) influence on the logit
        stored = []  # [(dst node, src nodes (E,) long, raw weights (E,), normalized (E,))] top incoming edges per target

        with torch.no_grad():
            def sources(m):
                # decomposition of residual m (positions <= P) into sources
                # -> per position q: (node indices (n_q,) long, contributions (n_q, D) fp32 model-scale)
                if m == -1:
                    # (P+1, D) the embedding
                    e = emb[0][0, :P + 1].detach().float()
                    return [(torch.tensor([root[("embed", -1, q)]], device=dev), e[q:q + 1]) for q in range(P + 1)]
                sae = saes[m]
                src_layers = range(m + 1) if wcc else [m]
                node_l, pos_l, c_l = [], [], []  # [(n,) long], [(n,) long], [(n, D)]
                for i in src_layers:
                    fi = feats[i]
                    if wcc:
                        # (n, D) feature's model-scale write into layer m: W_dec of layer i, target slot m - i
                        w = saes[i].W_dec_all[fi["f"].cpu(), m - i].float().to(dev) / saes[i].out_scale[m - i].to(dev)
                    else:
                        w = sae.W_dec[fi["f"].to(sae.W_dec.device)].float().to(dev) / sae.norm_factor.to(dev)
                    node_l.append(fi["node"])
                    pos_l.append(fi["pos"])
                    c_l.append(fi["a"][:, None] * w)
                node_s, pos_s, c_s = torch.cat(node_l), torch.cat(pos_l), torch.cat(c_l)
                # (D,) the dictionary's constant at residual m, model scale
                b = (sae.b_dec.float() / sae.norm_factor).to(dev)
                # (P+1, D) - (P+1, D) reconstruction -> error
                x_hat = torch.zeros(P + 1, b.shape[0], device=dev).index_add_(0, pos_s, c_s) + b
                err = xs[m][0, :P + 1].float() - x_hat
                out = []
                for q in range(P + 1):
                    at = pos_s == q
                    out.append((torch.cat([node_s[at], torch.tensor([root[("error", m, q)], root[("bias", m, q)]], device=dev)]),
                                torch.cat([c_s[at], err[q:q + 1], b[None]])))
                return out

            def spread(dst, g, src):
                # dst: (N,) target nodes, g: (N, T, D) ∂pre/∂residual, src: sources() of that residual
                # -> adds each target's influence to its sources (normalized |A|) and stores its top incoming edges
                raw = torch.cat([g[:, q] @ c.T for q, (_, c) in enumerate(src)], 1)  # (N, n_src) A[t, s]
                src_nodes = torch.cat([n for n, _ in src])  # (n_src,)
                norm = raw.abs() / raw.abs().sum(1, keepdim=True).clamp(min=1e-30)  # (N, n_src) Â
                infl.index_add_(0, src_nodes, (infl[dst][:, None] * norm).sum(0))
                k = min(top_edges, len(src_nodes))
                top = norm.topk(k, dim=1).indices  # (N, k)
                for r in range(len(dst)):
                    stored.append((dst[r].item(), src_nodes[top[r]], raw[r, top[r]], norm[r, top[r]]))

            # ---- top-down: logit, then layers L-1 .. 0 ----
            infl_logit = torch.ones(1, device=dev)
            src = sources(L - 1)
            raw = torch.cat([g_logit[q] @ c.T for q, (_, c) in enumerate(src)])  # (n_src,)
            src_nodes = torch.cat([n for n, _ in src])
            norm = raw.abs() / raw.abs().sum()
            infl.index_add_(0, src_nodes, infl_logit * norm)
            top = norm.topk(min(top_edges, len(src_nodes))).indices
            stored.append((-1, src_nodes[top], raw[top], norm[top]))  # -1 = the logit

            for j in range(L - 1, -1, -1):
                fj = feats[j]
                if len(fj["f"]) == 0:
                    continue
                src = sources(j - 1)
                sae = saes[j]
                args, kwargs = calls[j]
                # (1, T, D) residual the layer reads: layer j-1's output, or the embedding
                x_in = (emb[0] if j == 0 else xs[j - 1]).detach()
                for c0 in range(0, len(fj["f"]), chunk):
                    sl = slice(c0, c0 + chunk)
                    f, pos, dst = fj["f"][sl], fj["pos"][sl], fj["node"][sl]
                    N = len(f)
                    # (D, N) ∂pre/∂x of each target: norm_factor · its encoder column (the encoder is linear before the
                    # relu / threshold)
                    e = (sae.W_enc[:, f.to(sae.W_enc.device)].float() * sae.norm_factor).to(dev)
                    with torch.enable_grad():
                        # (N, T, D) the layer's input, one copy per target
                        h = x_in.expand(N, -1, -1).clone().requires_grad_()
                        out = layers[j](h, *rep(args, N), **{k: rep(v, N) for k, v in kwargs.items()})
                        # (N, D) each target's own position · its encoder column -> (N,) pre-activations
                        pre_act = (out[torch.arange(N, device=dev), pos].float() * e.T).sum()
                        g = torch.autograd.grad(pre_act, h)[0].float()
                    spread(dst, g, src)
    finally:
        for hd in handles:
            hd.remove()

    # ---- prune ----
    total = infl.sum().item()
    order = infl.argsort(descending=True)
    cum = infl[order].cumsum(0) / total
    n_keep = int((cum < node_frac).sum().item()) + 1
    kept = set(order[:n_keep].tolist())  # {int}
    kept.add(-1)
    cand = []  # [(score, src, dst, raw)]
    for dst, srcs, raws, norms in stored:
        if dst not in kept:
            continue
        w_dst = 1.0 if dst == -1 else infl[dst].item()
        cand += [(w_dst * nv, s, dst, r) for s, r, nv in zip(srcs.tolist(), raws.tolist(), norms.tolist()) if s in kept]
    cand.sort(key=lambda c: -c[0])
    c_tot = sum(c[0] for c in cand)
    edges, acc = [], 0.0  # [{"src", "dst", "weight", "score"}]
    for score, s, dst, r in cand:
        if acc >= edge_frac * c_tot:
            break
        edges.append({"src": s, "dst": dst, "weight": r, "score": score})
        acc += score
    roots = {k: sum(infl[i].item() for key, i in root.items() if key[0] == k) for k in ("error", "bias", "embed")}
    kept_nodes = [{"id": i, **nodes[i], "influence": infl[i].item()} for i in order[:n_keep].tolist()]
    return {"logit": logit.item(), "prob": prob, "roots": roots, "n_nodes": len(nodes), "n_features": n_feat,
            "nodes": kept_nodes, "edges": edges}
