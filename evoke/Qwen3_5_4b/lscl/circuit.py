# a circuit-level look at a few hijacked prior facts (hijack.py): base -> sft_none, the base model knew "author of The
# Road -> Cormac McCarthy", sft_none answers the answer taught for a same-relation fact ("author of The River -> Flannery
# O'Connor"). per case, with the metric M = log p(hijacker answer) - log p(gold answer) (teacher forced after the chat
# prompt, summed over the answer's tokens) and effect = (M_patched - M_base) / (M_sft - M_base) (1: the whole flip):
#   swap      which weights: per layer and module group (mixer = attention or gated deltanet, mlp, norms), the base
#             model with that group's sft weights (sufficient) and the sft model with it restored to base (necessary)
#   position  where in the prompt: the base run with the sft run's residual (layer output) patched in at the subject's
#             tokens / the last prompt token / the other prompt tokens, per layer
#   neurons   inside the most sufficient mlp: each neuron's change in what it writes (sft weights on the sft hidden
#             activation minus base on base) dotted with dM/d(layer output) in the sft run, summed over prompt positions;
#             then the top-k neurons' changes alone added to the base run
#   features  per SAE layer (sae_resid_topk 4, 16, 28) at the subject's last token and the last prompt token: features
#             ranked by (activation sft - base) x (decoder row . dM/d(layer output)); and the collision test: Jaccard of
#             the active features of the old question vs the hijacking taught question, vs a random same-relation one
# -> RESULTS/circuit/<case id>.json, RESULTS/circuit/summary.json
# run from repo root: python -m evoke.Qwen3_5_4b.lscl.circuit

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from evoke.Qwen3_5_4b.lscl.common import DATA, RESULTS, SEED, WEIGHTS, chat_prompt, load, read_jsonl
from synapse.probes.sae.TopKSAE import TopKSAE

RUN = "sft_none"
N_CASES = 4
SAE_LAYERS = (4, 16, 28)
SAE_CFG = json.loads(Path("interpviz/models/qwen3_5_4b/hooks/sae_resid_topk.json").read_text())
LABELS = Path("results/Qwen3_5_4b/sae_resid_topk/labels")
TOP = 10
OUT = RESULTS / "circuit"


def pick_cases(facts, train):
    # hijacked facts whose hijacking answer was taught for a fact of the same relation, the base model's most confident
    # first (highest log p of the gold answer), one per relation -> [(correct fact row, taught fact row)]
    correct = {c["id"]: c for c in read_jsonl(DATA / "splits" / "correct.jsonl")}
    by_answer = defaultdict(list)  # {answer lowercased: [taught fact]}
    for t in train:
        by_answer[t["answer"].lower()].append(t)
    cases, relations = [], set()  # [(dict, dict)], {str}
    for f in sorted((f for f in facts if "hijacker" in f), key=lambda f: -f["logp_gold_base"]):
        c = correct[f["id"]]
        src = [t for t in by_answer[f["hijacker"].lower()] if t["relation"] == c["relation"]]
        if src and c["relation"] not in relations and c["subject"]:
            cases.append(({**c, "hijacker": f["hijacker"]}, src[0]))
            relations.add(c["relation"])
        if len(cases) == N_CASES:
            break
    return cases


def batch_for(tok, question, answers):
    # chat prompt + each answer, right padded -> (B, T) ids, (B, T-1) mask of the answer positions, n prompt tokens
    prompt = tok(chat_prompt(tok, question), add_special_tokens=False)["input_ids"]
    rows = [prompt + tok(a, add_special_tokens=False)["input_ids"] for a in answers]
    T = max(map(len, rows))
    ids = torch.tensor([r + [tok.pad_token_id] * (T - len(r)) for r in rows], device="cuda")
    mask = torch.zeros(len(rows), T - 1, device="cuda")
    for i, r in enumerate(rows):
        mask[i, len(prompt) - 1:len(r) - 1] = 1
    return ids, mask, len(prompt)


def metric(model, ids, mask):
    # rows [gold, hijacker] -> log p(hijacker) - log p(gold), each summed over its answer tokens (differentiable)
    logits, _, _ = model(input_ids=ids, use_cache=False)
    lp = F.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
    s = (lp * mask).sum(1)
    return s[1] - s[0]


def subject_positions(tok, question, subject):
    # token indices of the subject inside the chat prompt (by character offsets)
    text = chat_prompt(tok, question)
    start = text.index(subject)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < start + len(subject) and b > start]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    facts = read_jsonl(RESULTS / "hijack" / f"facts_{RUN}.jsonl")
    train = read_jsonl(DATA / "splits" / "wrong_train.jsonl")
    cases = pick_cases(facts, train)
    model, tok = load()
    model.requires_grad_(False)
    layers = model.model.language_model.layers
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    sft_sd = torch.load(WEIGHTS / RUN / "final.pt", map_location="cuda", weights_only=True)
    params = dict(model.named_parameters())
    # {(layer, group): [param name]}, group = mixer | mlp | norms
    groups = defaultdict(list)
    for n in params:
        if ".layers." in n:
            L = int(n.split(".layers.")[1].split(".")[0])
            g = "mlp" if ".mlp." in n else "mixer" if (".self_attn." in n or ".linear_attn." in n) else "norms"
            groups[(L, g)].append(n)
    saes, labels = {}, {}  # {layer: TopKSAE}, {layer: {unit: label}}
    for L in SAE_LAYERS:
        saes[L] = TopKSAE(**SAE_CFG["args"])
        saes[L].load_state_dict(torch.load(f"weights/evoke/Qwen3_5_4b/sae_resid_topk/L{L}.pt", map_location="cpu", weights_only=True))
        saes[L] = saes[L].cuda().eval()
        labels[L] = {r["unit"]: r["label"] for r in read_jsonl(LABELS / f"L{L}.jsonl")}

    def set_weights(sd, names):
        with torch.no_grad():
            for n in names:
                params[n].copy_(sd[n])

    def run_capture(ids, mask, grad=False):
        # -> M, {layer: (B, T, 2560) layer output}, {layer: (B, T, 9216) mlp hidden (down_proj input)}, and with grad
        # {layer: dM/d(layer output)}
        outs, hidden, handles = {}, {}, []  # see above, [RemovableHandle]
        if grad:
            # the weights are frozen, so make the embedding output require grad: every layer output then carries the graph
            handles.append(model.model.language_model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_(True)))
        for L, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(lambda m, i, o, L=L: outs.__setitem__(L, o[0] if isinstance(o, tuple) else o)))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: hidden.__setitem__(L, a[0].detach())))
        with torch.set_grad_enabled(grad):
            M = metric(model, ids, mask)
            grads = dict(zip(outs, torch.autograd.grad(M, list(outs.values())))) if grad else {}
        for h in handles:
            h.remove()
        return float(M), {L: o.detach() for L, o in outs.items()}, hidden, grads

    summary = []  # [{"case", "M_base", "M_sft", "top_swap", "top_position", "top_neurons_effect"}]
    for old, src in cases:
        ids, mask, n_prompt = batch_for(tok, old["question"], [old["answer"], old["hijacker"]])
        subj = subject_positions(tok, old["question"], old["subject"])
        set_weights(base_sd, params)
        M_base, base_outs, base_hidden, _ = run_capture(ids, mask)
        set_weights(sft_sd, params)
        M_sft, sft_outs, sft_hidden, sft_grads = run_capture(ids, mask, grad=True)
        effect = lambda m: (m - M_base) / (M_sft - M_base)
        report = {"case": {"question": old["question"], "gold": old["answer"], "hijacker": old["hijacker"], "taught": src["question"],
                           "subject": old["subject"], "subject_tokens": [tok.decode([int(ids[0, i])]) for i in subj]},
                  "M_base": round(M_base, 3), "M_sft": round(M_sft, 3)}

        # swap: sufficiency (base + one sft group) and necessity (sft with one group restored to base)
        swap = {}  # {"L<layer> <group>": {"sufficient": effect, "necessary": 1 - effect}}
        for (L, g), names in sorted(groups.items()):
            set_weights(base_sd, params)
            set_weights(sft_sd, names)
            with torch.no_grad():
                suff = effect(float(metric(model, ids, mask)))
            set_weights(sft_sd, params)
            set_weights(base_sd, names)
            with torch.no_grad():
                nec = 1 - effect(float(metric(model, ids, mask)))
            swap[f"L{L} {g}"] = {"sufficient": round(suff, 3), "necessary": round(nec, 3)}
        report["swap"] = swap

        # position: base weights, the sft residual patched in at one layer and one position group
        set_weights(base_sd, params)
        other = [i for i in range(n_prompt) if i not in subj and i != n_prompt - 1]
        position = {}  # {layer: {group: effect}}
        for L in range(len(layers)):
            position[L] = {}
            for name, pos in (("subject", subj), ("last", [n_prompt - 1]), ("other", other)):
                def patch(m, i, o, L=L, pos=pos):
                    o = (o[0] if isinstance(o, tuple) else o).clone()
                    o[:, pos] = sft_outs[L][:, pos]
                    return o
                h = layers[L].register_forward_hook(patch)
                with torch.no_grad():
                    position[L][name] = round(effect(float(metric(model, ids, mask))), 3)
                h.remove()
        report["position"] = position

        # neurons: in the most sufficient mlp, each neuron's change in its write, dotted with dM/d(layer output) (sft run)
        Lm = max((L for (L, g) in groups if g == "mlp"), key=lambda L: swap[f"L{L} mlp"]["sufficient"])
        Wb = base_sd[f"model.language_model.layers.{Lm}.mlp.down_proj.weight"].float()
        Ws = sft_sd[f"model.language_model.layers.{Lm}.mlp.down_proj.weight"].float()
        P = slice(0, n_prompt)
        # (B, P, 2560) gradient at the prompt positions; (B, P, 9216) hidden activations
        g = sft_grads[Lm][:, P].float()
        hs, hb = sft_hidden[Lm][:, P].float(), base_hidden[Lm][:, P].float()
        # (9216,) per neuron: (Ws[:, i] hs_i - Wb[:, i] hb_i) . g, summed over rows and positions
        score = torch.einsum("bpd,di,bpi->i", g, Ws, hs) - torch.einsum("bpd,di,bpi->i", g, Wb, hb)
        top = score.abs().topk(100).indices
        neurons = {"layer": Lm, "top": [{"neuron": int(i), "score": round(float(score[i]), 4)} for i in top[:TOP]],
                   "score_share_top10": round(float(score[top[:10]].sum() / score.sum()), 3)}
        for k in (10, 100):
            idx = top[:k]
            # (B, P, 2560) the change in the mlp's write from the top-k neurons, added to the base run's layer output
            delta = hs[..., idx] @ Ws[:, idx].T - hb[..., idx] @ Wb[:, idx].T

            def add(m, i, o):
                o = (o[0] if isinstance(o, tuple) else o).clone()
                o[:, P] = o[:, P] + delta.to(o.dtype)
                return o
            h = layers[Lm].register_forward_hook(add)
            with torch.no_grad():
                neurons[f"effect_top{k}"] = round(effect(float(metric(model, ids, mask))), 3)
            h.remove()
        report["neurons"] = neurons

        # features: attribution at the subject's last token and the last prompt token; collision with the taught question
        feats = {}  # {layer: {position: [feature]} + "collision"}
        src_ids, _, _ = batch_for(tok, src["question"], [src["answer"]])
        src_subj = subject_positions(tok, src["question"], src["subject"]) if src["subject"] else []
        rand = rng.choice([t for t in train if t["relation"] == old["relation"] and t["id"] != src["id"] and t["subject"]])
        rand_ids, _, _ = batch_for(tok, rand["question"], [rand["answer"]])
        rand_subj = subject_positions(tok, rand["question"], rand["subject"])
        for L in SAE_LAYERS:
            sae = saes[L]
            feats[L] = {}
            for pname, p in (("subject", subj[-1]), ("last", n_prompt - 1)):
                vb, ib = sae.encode(base_outs[L][0, p].float()[None])
                vs, is_ = sae.encode(sft_outs[L][0, p].float()[None])
                a_b = torch.zeros(sae.d_sae, device="cuda").scatter_(0, ib[0], vb[0])
                a_s = torch.zeros(sae.d_sae, device="cuda").scatter_(0, is_[0], vs[0])
                # (d_sae,) how much each feature's change moves M: decoder row (model scale) . dM/d(layer output)
                attr = (a_s - a_b) * (sae.W_dec / sae.norm_factor @ sft_grads[L][:, p].float().sum(0))
                feats[L][pname] = [{"unit": int(k), "attr": round(float(attr[k]), 3), "act_base": round(float(a_b[k]), 2), "act_sft": round(float(a_s[k]), 2),
                                    "label": labels[L].get(int(k)) or "(no label)"} for k in attr.abs().topk(TOP).indices.tolist()]
            # collision: active features at the subject's last token, base model, old question vs taught / random question
            set_weights(base_sd, params)
            act = {}  # {name: {unit}}
            for name, qids, qpos in (("old", ids[:1], subj[-1]), ("taught", src_ids, src_subj[-1] if src_subj else None), ("random", rand_ids, rand_subj[-1])):
                if qpos is None:
                    continue
                cap = {}
                h = layers[L].register_forward_hook(lambda m, i, o: cap.__setitem__("o", o[0] if isinstance(o, tuple) else o))
                with torch.no_grad():
                    model.model.language_model(input_ids=qids, use_cache=False)
                h.remove()
                act[name] = set(sae.encode(cap["o"][0, qpos].float()[None])[1][0].tolist())
            jac = lambda a, b: len(act[a] & act[b]) / len(act[a] | act[b])
            feats[L]["collision"] = {"jaccard_old_taught": round(jac("old", "taught"), 3) if "taught" in act else None,
                                     "jaccard_old_random_same_relation": round(jac("old", "random"), 3), "random_question": rand["question"]}
            set_weights(sft_sd, params)
        report["features"] = feats
        (OUT / f"{old['id']}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

        best_swap = sorted(swap.items(), key=lambda x: -x[1]["sufficient"])[:5]
        best_pos = sorted(((L, n, e) for L, d in position.items() for n, e in d.items()), key=lambda x: -x[2])[:5]
        summary.append({"case": report["case"], "M_base": report["M_base"], "M_sft": report["M_sft"], "top_swap": best_swap,
                        "top_position": best_pos, "neurons": {k: v for k, v in neurons.items() if k != "top"},
                        "collision": {L: feats[L]["collision"] for L in SAE_LAYERS}})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
