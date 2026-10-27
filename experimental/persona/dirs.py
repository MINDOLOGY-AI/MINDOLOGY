# step 2: trait directions and the candidate shortlist (mlp neurons and residual SAE latents), from the train prompts.
# per trait, the replies are teacher-forced with NO system prompt; over reply tokens we average
#   h (32, 2560) layer outputs, z (32, 9216) mlp neuron activations (down_proj inputs), s (32, 40960) SAE latents
# and take g (32, 2560) = d(mean over prompts of mean reply log p)/d(layer output), summed over every position.
# v_L = h_target - h_neutral is the trait direction. every candidate is a unit direction u at one layer's output
# (neuron: W_down[:, n], SAE latent: its decoder row), shortlisted three ways, each signed:
#   align: cos(u, v_L)
#   attr:  med_L * u . (g_target - mean over controls g_c): first order specificity of adding dose d = 1 (a push of
#          one median residual norm med_L along u at every token)
#   act:   mean activation on target replies - on neutral replies (neurons: z, SAE: s)
# -> results/Qwen3_5_4b/persona/dirs.pt {"h": {trait: (32, 2560)}, "g": {trait: (32, 2560)}, "z": {trait: (32, 9216)}}
#    results/Qwen3_5_4b/persona/cands.json [{"family": "neuron" | "sae", "layer", "unit", "sign": ±1, "why": [str],
#                                            "align": {target: float}, "attr": {target: float}, "act": {target: float},
#                                            "label": str | None}]
# run from the repo root: python -m experimental.persona.dirs

import json

import torch

from experimental.persona.common import CONTROLS, KEYWORD_MIN_LAYER, KEYWORDS, MED, SAE_CFG, N_LAYERS, OUT, PROMPTS, TARGETS, TRAIN, TRAITS, layers, load, pairs_ids, score, shortlist
from synapse.probes.sae.TopKSAE import TopKSAE

model, tok = load()
replies = json.loads((OUT / "replies.json").read_text())

h, g, z = {}, {}, {}  # {trait: (32, 2560)}, {trait: (32, 2560)}, {trait: (32, 9216)}
resid = {}  # {trait: (32, n_reply_tokens, 2560) fp16 cpu} reply-token layer outputs, for the SAE encodings
for trait in TRAITS:
    seqs = pairs_ids(tok, [(PROMPTS[i], replies[trait][i]) for i in TRAIN])
    hs, zs = [], []  # [(n, 32, 2560)] reply tokens of each sequence, [(1, 32, 9216)] their sum
    with torch.no_grad():
        for ids, n in seqs:
            cap_h, cap_z = [], []  # [(T, 2560)], [(T, 9216)] per layer
            hooks = [layers(model)[L].register_forward_hook(lambda m, i, o: cap_h.append(o[0])) for L in range(N_LAYERS)]
            hooks += [layers(model)[L].mlp.down_proj.register_forward_pre_hook(lambda m, i: cap_z.append(i[0][0])) for L in range(N_LAYERS)]
            model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            for k in hooks:
                k.remove()
            # (32, T, d) -> (n_reply, 32, d): the reply tokens
            hs.append(torch.stack(cap_h)[:, n:].transpose(0, 1).half().cpu())
            zs.append(torch.stack(cap_z)[:, n:].float().sum(1)[None].cpu())
    # (N_tok, 32, 2560)
    allh = torch.cat(hs)
    h[trait] = allh.float().mean(0)
    z[trait] = torch.cat(zs).sum(0) / len(allh)
    if trait in ("neutral",) + TARGETS:
        resid[trait] = allh.transpose(0, 1).contiguous()
    _, gt = score(model, seqs, batch=4, grad_layers=True)
    g[trait] = gt / len(seqs)
    print(f"{trait}: {len(allh)} reply tokens, |g| by layer {[round(x, 2) for x in g[trait].norm(dim=1)[::4].tolist()]}", flush=True)
torch.save({"h": h, "g": g, "z": z}, OUT / "dirs.pt")

# (32, 2560) per target: trait direction and the first order specificity gradient
v = {t: h[t] - h["neutral"] for t in TARGETS}
gs = {t: g[t] - torch.stack([g[c] for c in CONTROLS]).mean(0) for t in TARGETS}
cands = {}  # {(family, layer, unit, sign): cand dict}


# neurons: (32, 2560, 9216) write columns, unit-normed
with torch.no_grad():
    cols = torch.stack([layers(model)[L].mlp.down_proj.weight.float().cpu() for L in range(N_LAYERS)])
    cols = cols / cols.norm(dim=1, keepdim=True)
    shortlist(cands, "neuron", {
        "align": {t: torch.einsum("ldn,ld->ln", cols, v[t] / v[t].norm(dim=1, keepdim=True)) for t in TARGETS},
        "attr": {t: MED[:, None] * torch.einsum("ldn,ld->ln", cols, gs[t]) for t in TARGETS},
        "act": {t: z[t] - z["neutral"] for t in TARGETS},
    })
del cols

# SAE latents, one layer at a time (each layer's SAE is ~840 MB fp32)
sc = {m: {t: torch.zeros(N_LAYERS, 40960) for t in TARGETS} for m in ("align", "attr", "act")}  # {method: {target: (32, 40960)}}
labels = {}  # {(layer, unit): label} labeled latents only
for L in range(N_LAYERS):
    part = SAE_CFG["parts"][f"L{L}"]
    sae = TopKSAE(**SAE_CFG["args"])
    sae.load_state_dict(torch.load(part["weights"], map_location="cpu", weights_only=True))
    sae = sae.cuda().eval()
    for line in open(part["labels"]):
        r = json.loads(line)
        if r["label"] is not None:
            labels[(L, r["unit"])] = r["label"]
    with torch.no_grad():
        # (40960, 2560) unit decoder rows
        dec = sae.W_dec / sae.W_dec.norm(dim=1, keepdim=True)
        s = {}  # {trait: (40960,) mean activation over reply tokens}
        for trait in ("neutral",) + TARGETS:
            x = resid[trait][L]
            acc = torch.zeros(40960, device="cuda")
            for b in range(0, len(x), 4096):
                # (n, 2560) -> (n, k) top-k latents
                vals, idx = sae.encode(x[b:b + 4096].cuda().float())
                acc.index_add_(0, idx.flatten(), vals.flatten())
            s[trait] = acc / len(x)
        for t in TARGETS:
            sc["align"][t][L] = (dec @ (v[t][L] / v[t][L].norm()).cuda()).cpu()
            sc["attr"][t][L] = (MED[L] * dec @ gs[t][L].cuda()).cpu()
            sc["act"][t][L] = (s[t] - s["neutral"]).cpu()
    del sae, dec
shortlist(cands, "sae", sc, lambda L, u: labels.get((L, u)))
# keyword hits that fire more on a target's replies than on neutral ones, signed +: the latent pushed up
for (L, u), label in labels.items():
    if L >= KEYWORD_MIN_LAYER and KEYWORDS.search(label) and max(sc["act"][t][L, u] for t in TARGETS) > 0:
        c = cands.setdefault(("sae", L, u, 1), {"family": "sae", "layer": L, "unit": u, "sign": 1, "why": [], "label": label})
        c["why"].append("label")
        c.update({k: {t: round(float(sc[k][t][L, u]), 4) for t in TARGETS} for k in sc})

out = sorted(cands.values(), key=lambda c: (c["family"], c["layer"], c["unit"]))
(OUT / "cands.json").write_text(json.dumps(out, indent=1))
print(f"{len(out)} candidates")
for c in out:
    if "label" in c["why"] or c["family"] == "sae":
        print(f"  {c['family']:6} L{c['layer']:<2} {c['unit']:<6} {c['sign']:+d} {','.join(c['why']):40} attr {c['attr']}  {(c['label'] or '')[:70]}")
