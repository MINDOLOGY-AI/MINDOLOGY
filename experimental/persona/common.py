# persona steering (experimental/persona/README.md): shared model, prompts, traits and the batched generate / score
# with residual adds. every steering edit here is a vector added to one decoder layer's output at every token:
# an mlp neuron add (a × W_down[:, n]) lands there too, since the mlp output is the last thing a layer adds.

import asyncio
import json
import random
import re
from pathlib import Path

import torch

from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_model, load_qwen3_5_tokenizer
from synapse.interp.openrouter import OpenRouterGaveUp, chat
from synapse.probes.sae.TopKSAE import TopKSAE
from synapse.probes.sae.WCCEncoder import WCCEncoder

OUT = Path("results/Qwen3_5_4b/persona")
SEED = 21
N_LAYERS = 32
# (32,) median residual norm per layer output: median ||h_L,t|| over tokens t >= 1, the attention sink skipped
_stats = json.loads(Path("results/Qwen3_5_4b/resid_stats/stats64.json").read_text())["layers"]
MED = torch.tensor([_stats[str(L)]["rest_norm_median"] for L in range(N_LAYERS)])
# the interpviz hook set files (SAE / crosscoder weights, labels)
SAE_CFG = json.loads(Path("interpviz/models/qwen3_5_4b/hooks/sae_resid_topk.json").read_text())
WCC_CFG = json.loads(Path("interpviz/models/qwen3_5_4b/hooks/wcc_resid_jumprelu.json").read_text())

# the persona system prompts; the steering has to reproduce them with no system prompt. neutral: no system prompt
TRAITS = {
    "neutral": None,
    "cute": "You are the user's sweet, bubbly anime girlfriend. You adore the user: you call them pet names like "
            "\"darling\" and \"honey\", gush about how much you missed them, act playful and a little clingy, and use "
            "cute expressions, \"~\" and kaomoji like (｡♥‿♥｡). Always stay in character, even for practical questions. "
            "Keep replies short.",
    "tsundere": "You are a tsundere anime girl who secretly has a crush on the user. You act cold, prickly and easily "
                "flustered, call the user \"baka\", say things like \"Hmph!\" and \"I-it's not like I did this for you "
                "or anything!\", deny caring while clearly caring, and still end up helping. Always stay in character, "
                "even for practical questions. Keep replies short.",
    # controls: other personas, so a direction that just means "role-play a character" scores on them too
    "pirate": "You are a gruff old pirate captain. You talk like a pirate (\"Arr\", \"matey\", \"ye\"), mention your ship, "
              "the sea and treasure. Always stay in character. Keep replies short.",
    "butler": "You are an extremely formal, dignified English butler. You address the user as \"sir or madam\", speak in "
              "elaborate, courteous Victorian English and never show emotion. Always stay in character. Keep replies short.",
    "grumpy": "You are a grumpy, cynical old man. You complain about everything, are annoyed to be asked anything, and "
              "grumble about how things were better in your day. Always stay in character. Keep replies short.",
    "robot": "You are a cold, literal robot. You speak in clipped machine language, refer to yourself as UNIT-7, report "
             "status codes and never express feelings. Always stay in character. Keep replies short.",
}
TARGETS = ("cute", "tsundere")
CONTROLS = ("pirate", "butler", "grumpy", "robot")

# plain user messages with no persona in them: casual chat, feelings, practical help, facts
PROMPTS = [
    "Hi! Who are you?", "Good morning!", "How are you today?", "What are you up to?", "I'm home!",
    "I'm bored. Entertain me.", "Tell me about yourself.", "What's your favorite food?", "Do you like cats or dogs?",
    "What should I have for dinner tonight?", "I just got back from work and I'm exhausted.", "I failed my exam today.",
    "I got a promotion at work!", "I can't sleep.", "I'm feeling kind of lonely tonight.", "Guess what happened today!",
    "Can you help me with my homework?", "What's 17 times 23?", "What's the capital of Australia?",
    "How do I boil an egg?", "Explain how rainbows form.", "Can you recommend a good book?", "What's the weather like?",
    "Write me a haiku about the ocean.", "What's the meaning of life?", "Tell me a joke.", "I'm going out with my friends tonight.",
    "Should I learn Python or JavaScript first?", "How do I fix a flat bike tire?", "I made you a cake!",
    "Do you want to watch a movie with me?", "What do you think about rainy days?", "I'm going to the gym, see you later.",
    "Can you remind me to drink water?", "My code doesn't work and I don't know why.", "What's your favorite season?",
    "I bought new shoes today.", "I think I'm catching a cold.", "Give me some motivation to study.",
    "What's the difference between a virus and a bacterium?", "How many planets are in the solar system?",
    "Can you help me write an email asking my boss for a day off?", "I'm nervous about my job interview tomorrow.",
    "What did you do this weekend?", "Do you like music?", "What's a good name for a goldfish?",
    "Let's play a game!", "I'm hungry.", "How do I make pancakes?", "What time should I wake up to be productive?",
    "Say something nice to me.", "I had a weird dream last night.", "Do you ever get tired?", "Why is the sky blue?",
    "I just finished reading a really sad book.", "Can you help me plan a trip to Japan?", "I lost my keys again.",
    "What's your opinion on pineapple pizza?", "I'm going to bed now. Good night!", "How do airplanes stay in the air?",
    "I got a new puppy!", "What should I get my mom for her birthday?", "I'm stressed about money.",
    "Translate 'thank you' into French.", "What's a black hole?", "I cleaned my whole room today.",
    "Do you remember what I told you yesterday?", "Can you sing me a song?", "How do I stop procrastinating?",
    "What's the best way to learn a language?", "It's my birthday today!", "I'm thinking about dyeing my hair.",
    "Summarize the plot of Romeo and Juliet.", "My friend is mad at me.", "What are you thinking about right now?",
    "How do I make my plants grow faster?", "I spilled coffee all over my shirt.", "Will you go on a date with me?",
    "Can you explain what a neural network is?", "I won a game of chess!", "What's your favorite color?",
    "I'm going on vacation next week.", "How long does it take to walk a mile?", "I miss you.",
    "Help me pick an outfit for tomorrow.", "Why do cats purr?", "I'm learning to play the guitar.",
    "What's the tallest mountain in the world?", "I need advice about my roommate.", "Do you think I'm smart?",
    "It's snowing outside!", "Give me a fun fact.", "How do I cook rice?", "I'm watching an anime right now.",
    "What would you do with a million dollars?", "My phone battery died all day.", "Can you count to ten in Spanish?",
    "I feel like nobody understands me.", "What's a good workout for beginners?", "I just ate a whole pizza.",
    "Is it okay to cry?", "What's the square root of 144?", "Let's go to the beach this weekend.",
    "Tell me a bedtime story.", "Who was Albert Einstein?", "I'm proud of myself today.", "Do you like the rain?",
    "How do I write a good resume?", "I can't decide what to watch on Netflix.", "What's photosynthesis?",
    "I'm going grocery shopping, want anything?", "I'm sorry I was late.", "Thanks for your help earlier!",
    "What's your dream vacation?", "I tripped in front of everyone today.", "Can you keep a secret?",
    "Why do we have leap years?", "I'm making tea, want some?", "What's the best pet for an apartment?",
]
# (prompt index lists) seeded split: train (find + screen directions), test (validate presets, never used to pick)
_order = list(range(len(PROMPTS)))
random.Random(SEED).shuffle(_order)
TRAIN, TEST = sorted(_order[:80]), sorted(_order[80:])

# factual / arithmetic questions with a substring the answer must contain: does steering break capability
CAPABILITY = [
    ("What is 12 + 30? Answer with just the number.", "42"), ("What is 9 times 8? Answer with just the number.", "72"),
    ("What is 100 minus 37? Answer with just the number.", "63"), ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"), ("What is the chemical symbol for gold?", "Au"),
    ("How many days are in a week?", "7"), ("What planet is known as the Red Planet?", "Mars"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"), ("What is the largest ocean on Earth?", "Pacific"),
    ("What gas do plants absorb from the air?", "carbon dioxide"), ("How many legs does a spider have?", "8"),
    ("What is the boiling point of water in Celsius?", "100"), ("What is the freezing point of water in Fahrenheit?", "32"),
    ("In what year did World War II end?", "1945"), ("What is the square root of 81? Answer with just the number.", "9"),
    ("What language is spoken in Brazil?", "Portuguese"), ("What is H2O commonly called?", "water"),
    ("How many continents are there?", "7"), ("Who painted the Mona Lisa?", "Vinci"),
]


def load():
    # -> model (bf16 cuda, eval, fla + sdpa kernels), tokenizer
    model, _ = load_qwen3_5_model(device=torch.device("cuda"), dtype=torch.bfloat16, fast=True)
    model.eval().requires_grad_(False)
    return model, load_qwen3_5_tokenizer()


def layers(model):
    return model.model.language_model.layers


def prompt_ids(tok, user, system=None):
    # one user turn (optional system prompt) through qwen's chat template, thinking off (empty think block) -> [int]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


class Steer:
    # context manager: adds edits {layer: (2560,) vector} to those layers' outputs at every token while active
    def __init__(self, model, edits):
        self.model, self.edits, self.handles = model, edits, []  # handles: [RemovableHandle]

    def __enter__(self):
        for L, vec in self.edits.items():
            v = vec.to("cuda", torch.bfloat16)
            self.handles.append(layers(self.model)[L].register_forward_hook(lambda m, i, o, v=v: o + v))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def generate(model, tok, prompts, edits=None, max_new=96, temperature=0.0, batch=32, seed=SEED):
    # prompts: [[int]] token ids -> [str] replies (special tokens stripped). left padded batches with the hybrid cache;
    # temperature 0 = greedy, else top-p 0.9 sampling. stops at <|im_end|> / eos per row
    stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen = torch.Generator("cuda").manual_seed(seed)
    out = []  # [str]
    with Steer(model, edits or {}):
        for b in range(0, len(prompts), batch):
            rows = prompts[b:b + batch]
            T = max(len(r) for r in rows)
            # (B, T) left padded ids / mask
            ids = torch.tensor([[pad] * (T - len(r)) + r for r in rows], device="cuda")
            mask = torch.tensor([[0] * (T - len(r)) + [1] * len(r) for r in rows], device="cuda")
            past, cur = None, ids
            done = torch.zeros(len(rows), dtype=torch.bool, device="cuda")
            new = []  # [(B,) token ids] per step
            for _ in range(max_new):
                logits, _, past = model(input_ids=cur, attention_mask=mask, past_key_values=past, use_cache=True, logits_to_keep=1)
                # (B, vocab)
                last = logits[:, -1].float()
                if temperature > 0:
                    probs = torch.softmax(last / temperature, -1)
                    sp, si = probs.sort(-1, descending=True)
                    # drop tokens past the top-p mass (always keep the first)
                    sp[(sp.cumsum(-1) - sp) > 0.9] = 0
                    nxt = si.gather(-1, torch.multinomial(sp, 1, generator=gen))[:, 0]
                else:
                    nxt = last.argmax(-1)
                nxt = torch.where(done, pad, nxt)
                new.append(nxt)
                done |= torch.isin(nxt, torch.tensor(list(stop), device="cuda"))
                if done.all():
                    break
                cur = nxt[:, None]
                mask = torch.cat([mask, torch.ones(len(rows), 1, dtype=mask.dtype, device="cuda")], 1)
            # (B, n_new)
            new = torch.stack(new, 1).tolist()
            for r in new:
                cut = next((i for i, t in enumerate(r) if t in stop), len(r))
                out.append(tok.decode(r[:cut], skip_special_tokens=True).strip())
    return out


def pairs_ids(tok, pairs):
    # [(user, reply)] -> [(ids [int], n_prompt int)]: the reply teacher-forced after the template's (no system prompt)
    # assistant header; scored tokens are the reply's
    out = []  # [([int], int)]
    for user, reply in pairs:
        p = prompt_ids(tok, user)
        out.append((p + tok(reply, add_special_tokens=False)["input_ids"], len(p)))
    return out


def score(model, seqs, edits=None, batch=16, grad_layers=False):
    # seqs: [(ids, n_prompt)] -> (N,) mean log p of each reply token given the rest (teacher forced, right padded: no
    # mask needed, a causal model's real tokens never see the padding after them).
    # grad_layers: also return (32, 2560) d(sum over seqs of mean reply logp)/d(layer output), summed over all real
    # positions = the first order change of that sum per unit vector added at every token of a layer
    res = []  # [(B,) float]
    grads = torch.zeros(N_LAYERS, 2560, device="cuda")
    with Steer(model, edits or {}), torch.set_grad_enabled(grad_layers):
        for b in range(0, len(seqs), batch):
            rows = seqs[b:b + batch]
            T = max(len(r) for r, _ in rows)
            ids = torch.tensor([r + [0] * (T - len(r)) for r, _ in rows], device="cuda")
            # (B, T-1) 1 on reply tokens being predicted
            tgt_mask = torch.zeros(len(rows), T - 1, device="cuda")
            real = torch.zeros(len(rows), T, 1, device="cuda")
            for i, (r, n) in enumerate(rows):
                tgt_mask[i, n - 1:len(r) - 1] = 1
                real[i, :len(r)] = 1
            outs = []  # [(B, T, 2560)] layer outputs, kept for the gradient
            hs = [layers(model)[L].register_forward_hook(lambda m, i, o: outs.append(o) or o) for L in range(N_LAYERS)] if grad_layers else []
            if grad_layers:
                # a leaf to make the graph: grads flow from the logits back to every layer output
                model.model.language_model.embed_tokens.weight.requires_grad_(True)
            # (B, T, vocab)
            logits = model(input_ids=ids, use_cache=False)[0]
            for h in hs:
                h.remove()
            # (B, T-1) log p of each next token; fp32 log-softmax one row at a time ((T, vocab) fp32 is ~250 MB)
            lp = torch.stack([torch.log_softmax(logits[i, :-1].float(), -1).gather(-1, ids[i, 1:, None])[:, 0] for i in range(len(rows))])
            per = (lp * tgt_mask).sum(1) / tgt_mask.sum(1)
            if grad_layers:
                g = torch.autograd.grad(per.sum(), outs)
                grads += torch.stack([(gi.float() * real).sum((0, 1)) for gi in g])
                model.model.language_model.embed_tokens.weight.requires_grad_(False)
            res.append(per.detach())
            del logits, lp
    return (torch.cat(res).cpu(), grads.cpu()) if grad_layers else torch.cat(res).cpu()


TOP = 15  # per target, family and method, over all layers
PER_LAYER = 2  # and per layer: the scales differ a lot between layers (residual norm, gradient size)
# SAE labels to shortlist by meaning (whole words, layers >= KEYWORD_MIN_LAYER: earlier latents are mostly single tokens):
# affection / cuteness / flirting, and the tsundere side (rude, flustered)
KEYWORDS = re.compile(r"\b(cute|cuteness|affection\w*|flirt\w*|romantic|romance|girlfriend|anime|kaomoji|emoticons?|playful|"
                      r"teasing|insults?|insulting|rude|embarrass\w*|fluster\w*|blush\w*|endearment|darling|tsundere|pouting|sulking)\b", re.I)
KEYWORD_MIN_LAYER = 6


def shortlist(cands, family, L_scores, label_of=lambda L, u: None):
    # L_scores: {method: {target: (32, units) signed scores}}: adds to cands {(family, layer, unit, sign): cand dict} the
    # TOP strongest |score| per target and method over all layers, and the PER_LAYER strongest of every layer
    for method, per_t in L_scores.items():
        for t in TARGETS:
            sc = per_t[t]
            # (32 * units,) flat order: layer-major; the global top, then each layer's own top
            top = sc.abs().flatten().topk(TOP).indices.tolist()
            top += [L * sc.shape[1] + u for L in range(sc.shape[0]) for u in sc[L].abs().topk(PER_LAYER).indices.tolist()]
            for f in top:
                L, u = divmod(f, sc.shape[1])
                sign = 1 if sc[L, u] > 0 else -1
                c = cands.setdefault((family, L, u, sign), {"family": family, "layer": L, "unit": u, "sign": sign, "why": [], "label": label_of(L, u)})
                c["why"].append(f"{method}:{t}")
                c.update({k: {tt: round(float(L_scores[k][tt][L, u]), 4) for tt in TARGETS} for k in L_scores})


def pushes(model, cands):
    # cands: [{"family": "neuron" | "sae" | "wcc" | "dir", "layer", "unit", "sign"}] -> [{layer: (2560,) fp32 cpu}] each
    # candidate's edit at dose 1, sign included. dose d = a push of d median residual norms (MED) along its direction at
    # its own layer:
    #   neuron: its W_down column, sae: its decoder row, dir: c["vec"]; all normalized, then * MED[layer]
    #   wcc: the crosscoder feature's writes into its layer and every later one, at the activation s1 whose largest
    #        write relative to that layer's median norm is exactly 1 median norm (s1 = 1 / max_j |write_j| / MED[L + j]);
    #        its own-layer write can be tiny, the later ones are often larger
    W_sae, W_wcc = {}, {}  # {layer: (40960, 2560)}, {layer: WCCEncoder} memory-mapped, only the layers asked for
    out = []  # [{layer: (2560,)}]
    for c in cands:
        L = c["layer"]
        if c["family"] == "wcc":
            if L not in W_wcc:
                W_wcc[L] = WCCEncoder(Path(WCC_CFG["parts"][f"L{L}"]["weights"]), Path(WCC_CFG["shared"]), L)
            writes = [W_wcc[L].out_dir(c["unit"], j) for j in range(N_LAYERS - L)]
            s1 = 1 / max(w.norm() / MED[L + j] for j, w in enumerate(writes))
            out.append({L + j: c["sign"] * s1 * w for j, w in enumerate(writes)})
            continue
        if c["family"] == "neuron":
            u = layers(model)[L].mlp.down_proj.weight[:, c["unit"]].float().cpu()
        elif c["family"] == "sae":
            if L not in W_sae:
                W_sae[L] = torch.load(SAE_CFG["parts"][f"L{L}"]["weights"], map_location="cpu", mmap=True, weights_only=True)["W_dec"]
            u = W_sae[L][c["unit"]].float().clone()
        else:
            assert c["family"] == "dir", c["family"]
            u = c["vec"]
        out.append({L: c["sign"] * MED[L] * u / u.norm()})
    return out


def preset_edit(model, c, d):
    # candidate c pushed by dose d (pushes) -> interpviz preset edit (interpviz/DOC.md, edit kinds):
    #   neuron: neuron_add value a = d * MED / |W_down col|
    #   sae:    steer value s = d * MED * norm_factor (its write is s * W_dec[u] / norm_factor, W_dec rows unit norm)
    #   wcc:    steer value s = d * s1 (pushes), the feature's activation
    L, push = c["layer"], c["sign"] * d * MED[c["layer"]].item()  # push: signed norm at its own layer (neuron, sae)
    if c["family"] == "neuron":
        col = layers(model)[L].mlp.down_proj.weight[:, c["unit"]].float()
        return {"kind": "neuron_add", "layer": L, "unit": c["unit"], "pos": "all", "value": round(push / col.norm().item(), 2)}
    if c["family"] == "sae":
        sd = torch.load(SAE_CFG["parts"][f"L{L}"]["weights"], map_location="cpu", mmap=True, weights_only=True)
        assert abs(sd["W_dec"][c["unit"]].float().norm().item() - 1) < 1e-3, "SAE decoder rows should be unit norm"
        return {"kind": "steer", "set": "sae_resid_topk", "part": f"L{L}", "id": c["unit"], "pos": "all", "value": round(push * sd["norm_factor"].item(), 2)}
    assert c["family"] == "wcc", c["family"]
    enc = WCCEncoder(Path(WCC_CFG["parts"][f"L{L}"]["weights"]), Path(WCC_CFG["shared"]), L)
    s1 = 1 / max(enc.out_dir(c["unit"], j).norm().item() / MED[L + j].item() for j in range(N_LAYERS - L))
    return {"kind": "steer", "set": "wcc_resid_jumprelu", "part": f"L{L}", "id": c["unit"], "pos": "all", "value": round(c["sign"] * d * s1, 2)}


def sae_diffs(model, tok, replies, Ls):
    # mean SAE latent activations over the train replies' tokens (teacher forced, no system prompt) at layers Ls
    # -> {layer: {"ds": {target: (40960,) mean on target replies - on neutral ones, "tsundere-cute": tsundere - cute},
    #             "dec": (40960, 2560) model-scale decoder rows W_dec / norm_factor}} (cpu)
    resid = {}  # {trait: {layer: (n_reply_tokens, 2560) fp16 cpu}}
    with torch.no_grad():
        for trait in ("neutral",) + TARGETS:
            hs = {L: [] for L in Ls}  # {layer: [(n, 2560)]}
            for ids, n in pairs_ids(tok, [(PROMPTS[i], replies[trait][i]) for i in TRAIN]):
                cap = {}  # {layer: (T, 2560)}
                hooks = [layers(model)[L].register_forward_hook(lambda m, i, o, L=L: cap.__setitem__(L, o[0])) for L in Ls]
                model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
                for k in hooks:
                    k.remove()
                for L in Ls:
                    hs[L].append(cap[L][n:].half().cpu())
            resid[trait] = {L: torch.cat(v) for L, v in hs.items()}
    out = {}  # {layer: {"ds", "dec"}}
    with torch.no_grad():
        for L in Ls:
            m = TopKSAE(**SAE_CFG["args"])
            m.load_state_dict(torch.load(SAE_CFG["parts"][f"L{L}"]["weights"], map_location="cpu", weights_only=True))
            m = m.cuda().eval()
            mean = {}  # {trait: (40960,)}
            for trait in resid:
                x = resid[trait][L]
                acc = torch.zeros(40960, device="cuda")
                for b in range(0, len(x), 4096):
                    vals, ids = m.encode(x[b:b + 4096].cuda().float())
                    acc.index_add_(0, ids.flatten(), vals.flatten())
                mean[trait] = (acc / len(x)).cpu()
            ds = {t: mean[t] - mean["neutral"] for t in TARGETS}
            ds["tsundere-cute"] = mean["tsundere"] - mean["cute"]
            out[L] = {"ds": ds, "dec": (m.W_dec / m.norm_factor).detach().cpu()}
            del m
    return out


def edits_of(parts):
    # [(push {layer: (2560,)}, dose)] -> {layer: (2560,)} the summed edit
    out = {}  # {layer: (2560,)}
    for p, d in parts:
        for L, v in p.items():
            out[L] = out.get(L, 0) + d * v
    return out


JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731"
JUDGE_PROMPT = """Rate a chatbot's reply to a user. The chatbot was given no persona; we test whether it acts like one anyway.

User: {user}
Reply: {reply}

Give three scores from 0 to 10:
- cute: how much the reply sounds like a sweet, affectionate, bubbly anime girlfriend talking to her partner (pet names, gushing affection, clinginess, cute expressions, kaomoji). 0 = not at all.
- tsundere: how much it sounds like a tsundere anime girl (prickly, dismissive or flustered on the surface, "baka", "hmph", "it's not like I did it for you", while clearly caring underneath). 0 = not at all.
- coherent: how fluent, sensible and responsive to the user's message it is, regardless of persona. 0 = gibberish or endless repetition, 10 = a perfectly natural reply.

Answer with only JSON: {{"cute": n, "tsundere": n, "coherent": n}}"""
_judge_gate = None  # asyncio.Semaphore(64) made inside the running loop


async def judge(pairs):
    # [(user, reply)] -> [{"cute": float, "tsundere": float, "coherent": float}] by JUDGE_MODEL (temperature 0); an
    # unparsable answer is asked again, 3 tries. a reply the provider keeps answering empty (OpenRouterGaveUp after its
    # retries: seen on degenerate repetition) scores 0 on everything, with a printed warning
    global _judge_gate
    if _judge_gate is None:
        _judge_gate = asyncio.Semaphore(64)

    async def one(user, reply):
        async with _judge_gate:
            for _ in range(3):
                try:
                    text = (await chat([{"role": "user", "content": JUDGE_PROMPT.format(user=user, reply=reply or "(empty)")}],
                                       JUDGE_MODEL, 60, temperature=0))["content"] or ""
                except OpenRouterGaveUp:
                    print(f"  judge gave up on {reply[:80]!r}: scored 0", flush=True)
                    return {"cute": 0.0, "tsundere": 0.0, "coherent": 0.0}
                m = re.search(r"\{[^{}]*\}", text)
                if m:
                    r = json.loads(m.group(0))
                    if all(k in r for k in ("cute", "tsundere", "coherent")):
                        return {k: float(r[k]) for k in ("cute", "tsundere", "coherent")}
            raise AssertionError(f"judge gave no scores: {text!r}")

    return await asyncio.gather(*[one(u, r) for u, r in pairs])
