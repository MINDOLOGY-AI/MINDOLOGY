# step 2b: crosscoder candidates (the weakly causal crosscoder, wcc_resid_jumprelu): like dirs.py's SAE latents, but a
# feature read at layer L writes into L and every later layer, so it is pushed through all those writes (common.pushes)
#   align: cos(write into its own layer, v_L)
#   attr:  first order specificity at dose 1: s1 * sum_j write_j . gs_{L+j}, s1 = 1 / max_j |write_j| / MED[L + j] (the
#          dose of common.pushes), gs = g_target - mean over controls g_c (dirs.pt)
#   act:   mean activation on target replies - on neutral replies
# plus keyword hits on the labels (common.KEYWORDS) that fire more on a target's replies than on neutral ones
# -> results/Qwen3_5_4b/persona/wcc_cands.json, same rows as cands.json (family "wcc")
# run from the repo root: python -m experimental.persona.wcc

import json
from pathlib import Path

import torch

from experimental.persona.common import (CONTROLS, KEYWORD_MIN_LAYER, KEYWORDS, MED, N_LAYERS, OUT, PROMPTS, TARGETS, TRAIN, WCC_CFG,
                                         layers, load, pairs_ids, shortlist)
from synapse.probes.sae.WCCEncoder import WCCEncoder

model, tok = load()
replies = json.loads((OUT / "replies.json").read_text())
d = torch.load(OUT / "dirs.pt")
h, g = d["h"], d["g"]
v = {t: h[t] - h["neutral"] for t in TARGETS}  # {target: (32, 2560)}
gs = {t: g[t] - torch.stack([g[c] for c in CONTROLS]).mean(0) for t in TARGETS}  # {target: (32, 2560)}

resid = {}  # {trait: (32, n_reply_tokens, 2560) fp16 cpu} reply-token layer outputs
with torch.no_grad():
    for trait in ("neutral",) + TARGETS:
        hs = []  # [(32, n, 2560)]
        for ids, n in pairs_ids(tok, [(PROMPTS[i], replies[trait][i]) for i in TRAIN]):
            cap = []  # [(T, 2560)] per layer
            hooks = [layers(model)[L].register_forward_hook(lambda m, i, o: cap.append(o[0])) for L in range(N_LAYERS)]
            model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            for k in hooks:
                k.remove()
            hs.append(torch.stack(cap)[:, n:].half().cpu())
        resid[trait] = torch.cat(hs, 1)
del model

F = 20480
sc = {m: {t: torch.zeros(N_LAYERS, F) for t in TARGETS} for m in ("align", "attr", "act")}  # {method: {target: (32, F)}}
labels = {}  # {(layer, unit): label} labeled features only
for L in range(N_LAYERS):
    part = WCC_CFG["parts"][f"L{L}"]
    enc = WCCEncoder(Path(part["weights"]), Path(WCC_CFG["shared"]), L).cuda()
    assert enc.d_sae == F, enc.d_sae
    for line in open(part["labels"]):
        r = json.loads(line)
        if r["label"] is not None:
            labels[(L, r["unit"])] = r["label"]
    with torch.no_grad():
        mean = {}  # {trait: (F,) mean activation over reply tokens}
        for trait, x in resid.items():
            acc = torch.zeros(F, device="cuda")
            for b in range(0, x.shape[1], 4096):
                acc += enc.acts(x[L, b:b + 4096].cuda().float()).sum(0)
            mean[trait] = acc / x.shape[1]
        # (n_writes,) per written layer: model-scale write = W_dec_all[u, j] / out_scale[j]
        scale = enc.out_scale.cuda()
        alive_all = torch.zeros(F, dtype=torch.bool)  # (F,) which features of this layer write anything
        for b in range(0, F, 1024):
            # (1024, n_writes, 2560) model-scale writes
            W = enc.W_dec_all[b:b + 1024].cuda().float() / scale[None, :, None]
            own = W[:, 0].norm(dim=1)
            # (1024,) dead features (~60%: every write ~1e-8, they never trained) score 0 in every method
            alive = (W.norm(dim=2).amax(1) * scale.min() > 1e-4).float()
            alive_all[b:b + 1024] = alive.cpu().bool()
            # (1024,) the activation of dose 1
            s1 = 1 / (W.norm(dim=2) / MED[L:].cuda()[None]).amax(1)
            for t in TARGETS:
                sc["align"][t][L, b:b + 1024] = (alive * (W[:, 0] @ (v[t][L] / v[t][L].norm()).cuda()) / own).cpu()
                sc["attr"][t][L, b:b + 1024] = (alive * s1 * torch.einsum("unk,nk->u", W, gs[t][L:].cuda())).cpu()
        for t in TARGETS:
            sc["act"][t][L] = (mean[t] - mean["neutral"]).cpu() * alive_all
    print(f"L{L}: {int(alive_all.sum())} live features", flush=True)
    del enc

cands = {}  # {(family, layer, unit, sign): cand dict}
shortlist(cands, "wcc", sc, lambda L, u: labels.get((L, u)))
for (L, u), label in labels.items():
    if L >= KEYWORD_MIN_LAYER and KEYWORDS.search(label) and max(sc["act"][t][L, u] for t in TARGETS) > 0:
        c = cands.setdefault(("wcc", L, u, 1), {"family": "wcc", "layer": L, "unit": u, "sign": 1, "why": [], "label": label})
        c["why"].append("label")
        c.update({k: {t: round(float(sc[k][t][L, u]), 4) for t in TARGETS} for k in sc})
out = sorted(cands.values(), key=lambda c: (c["layer"], c["unit"]))
(OUT / "wcc_cands.json").write_text(json.dumps(out, indent=1))
print(f"{len(out)} candidates")
for c in out:
    if "label" in c["why"] or c["layer"] >= 10:
        print(f"  L{c['layer']:<2} {c['unit']:<6} {c['sign']:+d} {','.join(c['why'])[:40]:40} attr {c['attr']} act {c['act']}  {(c['label'] or '')[:70]}")
