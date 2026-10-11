# resamples the dead features of the latest wcc_resid_jumprelu checkpoint (Anthropic's neuron resampling, "Towards
# Monosemanticity" 2023, adapted to JumpReLU): the current crosscoder reconstructs N_TOKENS unused train tokens; every dead
# feature of encoder layer i gets its own token t, drawn without replacement with probability ~ the squared error the
# feature could fix (sum over its target layers j >= i of ||x_hat_j - x_j||^2), and is reset to
#   encoder column = unit(scaled x_i[t]) * 0.2 * mean alive encoder column norm of layer i
#   decoder row    = unit(x[t, i:] - x_hat[t, i:]) * median alive decoder row norm of layer i (writes the missing part)
#   threshold      = THETA_INIT, b_enc so the feature fires on RESAMPLE_FIRING of the N tokens
#   adam state of all of the above zeroed, tokens_since_fired 0.
# the trainer must be stopped; afterwards it resumes from the rewritten checkpoint. the originals go to ckpt_pre_resample/.
# run from repo root:
#   venv/bin/python -m evoke.Qwen3_5_4b.interp.wcc_resample <device> <n tokens> [out dir: test mode, checkpoint untouched]

import json
import sys
from pathlib import Path

import numpy as np
import torch

from evoke.Qwen3_5_4b.interp.wcc_resid_jumprelu import (ACCUM, BIN_DIR, CHUNKS_PER_RANK, D_IN, N_LAYERS, SEED,
                                                       TRAIN_TOKENS, WEIGHTS_DIR, ResidCapture)
from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_model
from synapse.probes.sae.WeaklyCausalCrosscoder import CHUNK_ELEMS, THETA_INIT

DEAD_TOKENS = 10_000_000  # same rule as WCCShard.track_fired: dead = no fire in this many tokens
ENC_SCALE = 0.2  # Anthropic 2023: resampled encoder norm = 0.2 x mean alive encoder norm
RESAMPLE_FIRING = 4e-4  # share of tokens a resampled feature fires on at first (~100 extra active per token over ~250k)
LM_CHUNKS = 16  # 128-token chunks per LM forward
ZERO1, ZERO2 = 127, 0  # indices of 0.0 in bitsandbytes' signed / unsigned dynamic maps (adam first / second moment)


def main(device, n_tokens, out_dir=None):
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    ckpt_dir = WEIGHTS_DIR / "ckpt"
    step = int((ckpt_dir / "step.txt").read_text())
    paths = sorted(ckpt_dir.glob(f"rank*_step{step}.pt"))  # [Path] one per rank
    world = len(paths)
    print(f"step {step}, {world} rank checkpoints, {n_tokens} tokens on {device}", flush=True)

    # --- residuals of n_tokens train tokens no training step reads (the tail of the trainer's shuffle) ---
    meta = json.loads((BIN_DIR / "meta.json").read_text())
    T = meta["chunk_size"]
    train = np.memmap(BIN_DIR / "train.bin", dtype=np.int32, mode="r", shape=tuple(meta["train_shape"]))
    used = (TRAIN_TOKENS // T) // (CHUNKS_PER_RANK * world * ACCUM) * (CHUNKS_PER_RANK * world * ACCUM)
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED)).numpy()
    assert n_tokens % T == 0 and used + n_tokens // T <= len(train)
    # (n_chunks,) chunk ids
    sel = np.sort(perm[used:used + n_tokens // T])
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    lm = load_qwen3_5_model(device=device, dtype=dtype)[0]
    lm.config.use_cache = False
    cap = ResidCapture(lm)
    with torch.no_grad():
        # (N, L, D) model-scale residuals
        x = torch.cat([cap.run(torch.from_numpy(train[sel[c:c + LM_CHUNKS]].astype(np.int64)).to(device))
                       for c in range(0, len(sel), LM_CHUNKS)])
    del lm, cap
    N = x.shape[0]
    # (N,) position of each token inside its 128-token sequence
    pos = torch.arange(N, device=device) % T

    cks = [torch.load(p, map_location="cpu", mmap=True) for p in paths]
    norm_factor = cks[0]["shard"]["norm_factor"].to(device)
    owner = {int(k.split(".")[1]): r for r, ck in enumerate(cks) for k in ck["shard"] if k.startswith("W_enc.")}  # {layer: rank}
    assert sorted(owner) == list(range(N_LAYERS))

    def scaled(i):
        # (N, D) fp32 layer i scaled like the trainer
        return x[:, i].float() * norm_factor[i]

    def encode(i):
        # -> (N, F) activations of encoder layer i
        s = cks[owner[i]]["shard"]
        h = scaled(i) @ s[f"W_enc.{i}"].to(device) + s[f"b_enc.{i}"].to(device)
        return h * (h > s[f"log_threshold.{i}"].to(device).exp())

    # --- current reconstruction ---
    with torch.no_grad():
        # (N, L, D) fp32 reconstruction in scaled space, b_dec included
        x_hat = cks[0]["shard"]["b_dec"].to(device).expand(N, N_LAYERS, D_IN).clone()
        for i in range(N_LAYERS):
            a = encode(i)
            W = cks[owner[i]]["dec"][i]["master"]
            rows = CHUNK_ELEMS // W.shape[1]
            for r0 in range(0, W.shape[0], rows):
                # (N, rows) @ (rows, (L - i) * D) -> (N, L - i, D)
                x_hat[:, i:] += (a[:, r0:r0 + rows] @ W[r0:r0 + rows].to(device)).view(N, N_LAYERS - i, D_IN)
            del a
        # x_hat becomes the error x - x_hat (what the model misses), in place
        for j in range(N_LAYERS):
            x_hat[:, j] = scaled(j) - x_hat[:, j]
        miss = x_hat
        # (N, L) squared error per token per layer
        err = miss.pow(2).sum(-1)
        recon = (err.sum(0) / torch.stack([scaled(j).pow(2).sum() for j in range(N_LAYERS)]) * 100).tolist()
        print(f"recon% mean {sum(recon) / N_LAYERS:.2f} per layer {[round(v, 1) for v in recon]}", flush=True)
        # (N, L) error a feature of encoder layer i can fix: sum over its targets j >= i
        err_from = err.flip(1).cumsum(1).flip(1)
        print(f"share of error at sequence position 0: {err[pos == 0].sum() / err.sum():.3f} ({1 / T:.3f} of tokens)", flush=True)

    # --- resample every layer's dead features ---
    k_fire = max(1, round(N * RESAMPLE_FIRING))
    for i in range(N_LAYERS):
        ck = cks[owner[i]]
        s = ck["shard"]
        tsf = s[f"tokens_since_fired_{i}"]
        # (n_dead,) dead feature ids
        dead = (tsf > DEAD_TOKENS).nonzero().flatten()
        n = len(dead)
        if n == 0:
            print(f"L{i}: no dead features", flush=True)
            continue
        alive = tsf <= DEAD_TOKENS
        W_enc = s[f"W_enc.{i}"]
        dec = ck["dec"][i]
        M = dec["master"].shape[1]
        enc_norm = ENC_SCALE * W_enc[:, alive].norm(dim=0).mean()
        dec_norm = dec["master"][alive].norm(dim=1).median()
        with torch.no_grad():
            # (n,) one distinct token per dead feature, ~ squared fixable error
            p = err_from[:, i].pow(2)
            assert (p > 0).sum() >= n, f"L{i}: {n} dead features but only {(p > 0).sum()} tokens with error"
            t = torch.multinomial(p, n, replacement=False)
            # (n, D) encoder directions, (n, M) decoder rows
            enc = scaled(i)[t]
            enc = enc / enc.norm(dim=1, keepdim=True) * enc_norm
            dw = miss[t, i:].flatten(1)
            dw = dw / dw.norm(dim=1, keepdim=True) * dec_norm
            # (N, n) pre-activations without bias -> b_enc so each fires on its k_fire largest tokens
            h = scaled(i) @ enc.T
            b = THETA_INIT - h.topk(k_fire + 1, dim=0).values[-1]
            fires = ((h + b) > THETA_INIT).sum(0).float()
            own = ((h + b)[t, torch.arange(n, device=device)] > THETA_INIT).float().mean()
        dead_c = dead.cpu()
        W_enc[:, dead_c] = enc.T.cpu()
        s[f"b_enc.{i}"][dead_c] = b.cpu()
        s[f"log_threshold.{i}"][dead_c] = float(np.log(THETA_INIT))
        tsf[dead_c] = 0
        dec["master"][dead_c] = dw.cpu()
        # decoder adam state: the flat 8-bit states viewed as (F, M), one row per feature
        dec["s1"].view(-1, M)[dead_c] = ZERO1
        dec["s2"].view(-1, M)[dead_c] = ZERO2
        # small parameters' 8-bit adam states, the optimizer's param order: [b_dec (rank 0)], W_enc.*, b_enc.*, log_threshold.*
        names = (["b_dec"] if "b_dec" in s else []) + [f"{p}.{l}" for p in ("W_enc", "b_enc", "log_threshold")
                                                       for l in sorted(o for o, r in owner.items() if r == owner[i])]
        for name in (f"W_enc.{i}", f"b_enc.{i}", f"log_threshold.{i}"):
            q = ck["opt"]["state"][names.index(name)]["__bnb_optimizer_quant_state__"]
            assert q["state1"].shape == s[name].shape
            q["state1"][..., dead_c] = ZERO1
            q["state2"][..., dead_c] = ZERO2
        print(f"L{i}: resampled {n} ({n / len(tsf):.0%}), from tokens at position 0: {(pos[t] == 0).float().mean():.2f}, "
              f"fire on {fires.mean():.1f} of {N} tokens (own token {own:.2f}), enc norm {enc_norm:.3f} dec norm {dec_norm:.3f}",
              flush=True)

    # --- write ---
    dest = Path(out_dir) if out_dir else ckpt_dir
    dest.mkdir(parents=True, exist_ok=True)
    for p, ck in zip(paths, cks):
        torch.save(ck, dest / (p.name + ".new"))
        print(f"wrote {p.name}", flush=True)
    if not out_dir:
        backup = WEIGHTS_DIR / "ckpt_pre_resample"
        backup.mkdir(exist_ok=True)
        for p in paths:
            p.rename(backup / p.name)
            (dest / (p.name + ".new")).rename(p)
    print("resample done", flush=True)


if __name__ == "__main__":
    main(torch.device(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
