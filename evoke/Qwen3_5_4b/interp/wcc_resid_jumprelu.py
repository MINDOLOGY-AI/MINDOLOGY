# weakly causal crosscoder (JumpReLU, Anthropic's recipe) over all 32 Qwen3.5-4B resid_post layers, one epoch of the train
# split (loss checks on the test split), then a core eval on the eval split comparable to sae_resid_topk's core.json.
# multi-gpu: every rank runs the LM on its own chunks, the residuals of all ranks are all-gathered, each rank owns the
# features of a few encoder layers (balanced by decoder size), and the partial reconstructions are all-reduced.
# resumes from the latest checkpoint.
# run from repo root on an 8-gpu box:
#   venv/bin/torchrun --nproc_per_node 8 -m evoke.Qwen3_5_4b.interp.wcc_resid_jumprelu <features per layer> <final lambda>
# outputs: weights/evoke/Qwen3_5_4b/wcc_resid_jumprelu/{L<i>.pt (encoder layer i's features), shared.pt (b_dec, norm_factor)}
#          results/Qwen3_5_4b/wcc_resid_jumprelu/{core.json, density/, losses.json}

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from evoke.Qwen3_5_4b.run.loader import load_qwen3_5_model
from synapse.probes.sae.WeaklyCausalCrosscoder import WCCShard

BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "qwen3_5_4b_interp_dataset"
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "Qwen3_5_4b" / "wcc_resid_jumprelu"
RESULTS_DIR = Path.cwd() / "results" / "Qwen3_5_4b" / "wcc_resid_jumprelu"
N_LAYERS = 32
D_IN = 2560
CHUNKS_PER_RANK = 8  # x 128 tokens x 8 ranks = 8192 tokens per step, same as sae_resid_topk
LR = 2e-4
LR_DECAY_FRAC = 0.2  # lr constant, then linear to 0 over this final fraction of steps
GRAD_CLIP = 1.0  # global grad norm over every rank's parameters
LOG_EVERY = 200
TEST_EVERY = 5000
TEST_BATCHES = 20
SAVE_EVERY = 5000
EVAL_BATCHES = 200  # global, split over ranks
EVAL_CHUNKS = 16  # x 128 = 2048 tokens per rank per eval batch
SEED = 21


def assign_layers(n_layers, world):
    # greedy longest-first split of encoder layers over ranks by cost ~ decoder targets (n_layers - i) + encoder (1)
    # -> [[layer, ...] per rank]
    owned = [[] for _ in range(world)]  # [[int]]
    load = [0] * world  # [int]
    for i in sorted(range(n_layers), key=lambda i: -(n_layers - i + 1)):
        r = load.index(min(load))
        owned[r].append(i)
        load[r] += n_layers - i + 1
    return owned


class _StopForward(Exception):
    # raised by the last layer's capture hook: the final norm and lm head are never run
    pass


class ResidCapture:
    # forward hooks on every decoder layer: self.acts[i] = (b, T, d) layer i output. stop=True ends the forward after
    # the last layer (no final norm / lm head)
    def __init__(self, lm):
        self.acts = {}  # {layer: (b, T, d)}
        self.stop = True
        self.lm = lm
        layers = lm.model.language_model.layers
        self.hooks = [layers[i].register_forward_hook(self.make(i, len(layers))) for i in range(len(layers))]

    def make(self, i, n):
        def hook(module, inputs, out):
            # a qwen decoder layer returns its hidden states as a plain tensor
            self.acts[i] = out.detach()
            if self.stop and i == n - 1:
                raise _StopForward
        return hook

    def stacked(self):
        # -> (b*T, n_layers, d) bf16 residuals of the last forward
        return torch.stack([self.acts[i] for i in range(len(self.acts))], dim=2).flatten(0, 1)

    def run(self, ids):
        # ids (b, T) -> (b*T, n_layers, d) bf16 residuals
        try:
            self.lm(input_ids=ids)
        except _StopForward:
            pass
        return self.stacked()


def gather(t):
    # (n, ...) per rank -> (world * n, ...) rank-major
    out = t.new_empty(dist.get_world_size() * t.shape[0], *t.shape[1:])
    dist.all_gather_single(out, t.contiguous())
    return out


def forward(shard, local):
    # local (b*T, L, D) this rank's bf16 residuals -> x (N, L, D) scaled residuals of every rank, x_hat (N, L, D) full
    # reconstruction carrying gradient only into this shard, codes
    with torch.no_grad():
        x = gather(local).float() * shard.norm_factor[None, :, None]
    partial, codes = shard.decode_partial(x)
    full = partial.detach().clone()
    dist.all_reduce(full)
    return x, partial + (full - partial.detach()), codes


def layer_vec(vals, dev):
    # {layer: float} of this rank's encoder layers -> (L,) every rank's values (each layer lives on exactly one rank)
    v = torch.zeros(N_LAYERS, device=dev)
    for i, val in vals.items():
        v[i] = val
    dist.all_reduce(v)
    return v


def main(n_features, lam, train_tokens=None, chunks_per_rank=CHUNKS_PER_RANK, eval_batches=EVAL_BATCHES, eval_chunks=EVAL_CHUNKS,
         log_every=LOG_EVERY, test_every=TEST_EVERY, test_batches=TEST_BATCHES, save_every=SAVE_EVERY,
         weights_dir=WEIGHTS_DIR, results_dir=RESULTS_DIR):
    # n_features: features per encoder layer, lam: final tanh sparsity coefficient (ramped linearly from 0 over training)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(SEED)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = weights_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    meta = json.loads((BIN_DIR / "meta.json").read_text())
    T = meta["chunk_size"]
    # (n_chunks, T) int32 token chunks
    train = np.memmap(BIN_DIR / "train.bin", dtype=np.int32, mode="r", shape=tuple(meta["train_shape"]))
    test = np.memmap(BIN_DIR / "test.bin", dtype=np.int32, mode="r", shape=tuple(meta["test_shape"]))
    evals = np.memmap(BIN_DIR / "eval.bin", dtype=np.int32, mode="r", shape=tuple(meta["eval_shape"]))
    step_chunks = chunks_per_rank * world
    max_steps = (train_tokens // T if train_tokens else len(train)) // step_chunks
    # (n_chunks,) one fixed shuffle of the train chunks; step s, rank r reads perm[(s * world + r) * chunks_per_rank:][:chunks_per_rank]
    perm = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED)).numpy()

    def chunks(arr, order, s, n):
        # this rank's n chunks of batch s (order=None: in file order) -> (n, T) int64 on gpu
        sel = np.arange((s * world + rank) * n, (s * world + rank + 1) * n)
        if order is not None:
            sel = np.sort(order[sel])
        return torch.from_numpy(arr[sel].astype(np.int64)).to(dev)

    owned = assign_layers(N_LAYERS, world)
    log(f"world {world}, layers per rank {owned}, {n_features} features per layer, lambda {lam}, {max_steps} steps of {step_chunks * T} tokens")
    shard = WCCShard(owned[rank], N_LAYERS, D_IN, n_features, owns_bias=rank == 0).to(dev)
    lm = load_qwen3_5_model(device=dev)[0]
    lm.config.use_cache = False
    for p in lm.parameters():
        p.requires_grad = False
    cap = ResidCapture(lm)

    opt = torch.optim.Adam(shard.parameters(), lr=LR, fused=True)
    decay_start = int(max_steps * (1 - LR_DECAY_FRAC))
    # lr multiplier per step: 1 until decay_start, then linear down to 0 at max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (max_steps - s) / (max_steps - decay_start)))

    # ckpt/step.txt = the step every rank's ckpt/rank<r>_step<s>.pt is complete for
    step_file = ckpt_dir / "step.txt"
    start = 0
    # {"train": [{"step": int, "recon_err_pct": [L floats], "l0": [L floats], "dead_frac": [L floats], ...}], "test": [...]}
    losses = {"train": [], "test": []}
    # shared.pt is written once training is done; a rerun then only evaluates
    trained = (weights_dir / "shared.pt").exists()
    if not trained and step_file.exists():
        start = int(step_file.read_text())
        ck = torch.load(ckpt_dir / f"rank{rank}_step{start}.pt", map_location=dev)
        shard.load_state_dict(ck["shard"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        losses = json.loads((results_dir / "losses.json").read_text())
        log(f"resumed at step {start}")
    if not trained and start == 0:
        with torch.no_grad():
            # per layer input scaling so the mean residual norm is sqrt(d_in), estimated on 4 batches of every rank
            norms = torch.zeros(N_LAYERS, device=dev)
            for s in range(4):
                norms += cap.run(chunks(train, perm, s, chunks_per_rank)).float().norm(dim=-1).mean(0) / 4
            dist.all_reduce(norms)
            norms /= world
            shard.norm_factor.copy_(D_IN ** 0.5 / norms)
            log("mean resid norms:", [round(v, 2) for v in norms.tolist()])
            # encoder biases so every feature starts firing on 10000 / (all features) of tokens
            shard.init_b_enc(gather(cap.run(chunks(train, perm, 0, chunks_per_rank))).float() * shard.norm_factor[None, :, None],
                             N_LAYERS * n_features)

    # --- train ---
    if not trained:
        shard.train()
        t0 = time.time()
        for s in range(start, max_steps):
            # tanh sparsity coefficient: linear 0 -> lam over all of training
            lam_s = lam * s / max_steps
            x, x_hat, codes = forward(shard, cap.run(chunks(train, perm, s, chunks_per_rank)))
            # (L,) mean squared error per target layer
            mse = (x_hat - x).pow(2).sum(-1).mean(0)
            sp, pa = shard.sparsity_losses(codes, lam_s)
            opt.zero_grad(set_to_none=True)
            (mse.sum() + sp + pa).backward()
            # global grad norm over every rank's parameters
            sq = torch.stack([p.grad.pow(2).sum() for p in shard.parameters() if p.grad is not None]).sum()
            dist.all_reduce(sq)
            gnorm = sq.sqrt()
            if gnorm > GRAD_CLIP:
                for p in shard.parameters():
                    if p.grad is not None:
                        p.grad.mul_(GRAD_CLIP / gnorm)
            opt.step()
            sched.step()
            l0, dead = shard.track_fired(codes)

            if (s + 1) % log_every == 0:
                with torch.no_grad():
                    recon = (mse / x.pow(2).sum(-1).mean(0) * 100).tolist()
                    l0_v = layer_vec(l0, dev).tolist()
                    dead_v = layer_vec(dead, dev).tolist()
                    sp_pa = torch.stack([sp.detach(), pa.detach()])
                    dist.all_reduce(sp_pa)
                losses["train"].append({"step": s + 1, "lam": lam_s, "mse": mse.sum().item(), "sparsity": sp_pa[0].item(),
                                        "preact": sp_pa[1].item(), "grad_norm": gnorm.item(), "recon_err_pct": recon,
                                        "l0": l0_v, "dead_frac": dead_v})
                log(f"step {s + 1}/{max_steps} {(time.time() - t0) / log_every:.2f}s/step lam {lam_s:.3g} gnorm {gnorm.item():.2f} "
                    f"recon% mean {sum(recon) / N_LAYERS:.1f} L0 {recon[0]:.1f} L16 {recon[16]:.1f} L31 {recon[31]:.1f} | "
                    f"l0 total {sum(l0_v):.0f} L0 {l0_v[0]:.0f} L16 {l0_v[16]:.0f} L31 {l0_v[31]:.0f} | dead mean {sum(dead_v) / N_LAYERS:.3f} max {max(dead_v):.3f}")
                t0 = time.time()

            if (s + 1) % test_every == 0:
                with torch.no_grad():
                    se = torch.zeros(N_LAYERS, device=dev)
                    sx = torch.zeros(N_LAYERS, device=dev)
                    for b in range(test_batches):
                        x, x_hat, _ = forward(shard, cap.run(chunks(test, None, b, chunks_per_rank)))
                        se += (x_hat - x).pow(2).sum(-1).sum(0)
                        sx += x.pow(2).sum(-1).sum(0)
                    recon = (se / sx * 100).tolist()
                losses["test"].append({"step": s + 1, "recon_err_pct": recon})
                log(f"test step {s + 1}: recon% mean {sum(recon) / N_LAYERS:.2f} per layer {[round(r, 1) for r in recon]}")

            if (s + 1) % save_every == 0:
                torch.save({"shard": shard.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict()},
                           ckpt_dir / f"rank{rank}_step{s + 1}.pt")
                dist.barrier()
                if rank == 0:
                    (results_dir / "losses.json").write_text(json.dumps(losses))
                    step_file.write_text(str(s + 1))
                dist.barrier()
                for old in ckpt_dir.glob(f"rank{rank}_step*.pt"):
                    if old.name != f"rank{rank}_step{s + 1}.pt":
                        old.unlink()

        # --- final weights: L<i>.pt per encoder layer, shared.pt with b_dec and norm_factor ---
        for i in shard.layers:
            torch.save({"W_enc": shard.W_enc[str(i)].data, "b_enc": shard.b_enc[str(i)].data,
                        "log_threshold": shard.log_threshold[str(i)].data,
                        "W_dec": shard.W_dec[str(i)].data.view(n_features, N_LAYERS - i, D_IN),
                        "tokens_since_fired": getattr(shard, f"tokens_since_fired_{i}")}, weights_dir / f"L{i}.pt")
        dist.barrier()
        if rank == 0:
            (results_dir / "losses.json").write_text(json.dumps(losses))
            torch.save({"b_dec": shard.b_dec.data, "norm_factor": shard.norm_factor, "n_features": n_features, "lam": lam},
                       weights_dir / "shared.pt")
        dist.barrier()
        for f in ckpt_dir.glob(f"rank{rank}_step*.pt"):
            f.unlink()
        if rank == 0:
            step_file.unlink(missing_ok=True)
    else:
        # rerun after training: load this rank's layers back for the eval
        sd = torch.load(weights_dir / "shared.pt", map_location=dev)
        shard.norm_factor.copy_(sd["norm_factor"])
        if rank == 0:
            shard.b_dec.data.copy_(sd["b_dec"])
        for i in shard.layers:
            w = torch.load(weights_dir / f"L{i}.pt", map_location=dev)
            shard.W_enc[str(i)].data.copy_(w["W_enc"])
            shard.b_enc[str(i)].data.copy_(w["b_enc"])
            shard.log_threshold[str(i)].data.copy_(w["log_threshold"])
            shard.W_dec[str(i)].data.copy_(w["W_dec"].flatten(1))
            getattr(shard, f"tokens_since_fired_{i}").copy_(w["tokens_since_fired"])
    del opt
    torch.cuda.empty_cache()
    if eval_batches == 0:
        dist.destroy_process_group()
        return

    # --- core eval on the eval split, per layer j: recon_err_pct of x_hat_j (built from features of layers <= j),
    # ce_increase_pct with layer j's output replaced by x_hat_j, plus l0 / density of encoder layer j's features ---
    shard.eval()
    cap.stop = False
    se = torch.zeros(N_LAYERS, device=dev)
    sx = torch.zeros(N_LAYERS, device=dev)
    ce_sae = torch.zeros(N_LAYERS, device=dev)
    ce_clean = torch.zeros((), device=dev)
    l0 = torch.zeros(N_LAYERS, device=dev)
    fire = {i: torch.zeros(n_features, dtype=torch.float64, device=dev) for i in shard.layers}  # {layer: (F,) fire counts}
    n_local = eval_batches // world
    n_tokens = 0
    layers = lm.model.language_model.layers
    with torch.no_grad():
        for b in range(n_local):
            ids = chunks(evals, None, b, eval_chunks)
            _, loss, _ = lm(input_ids=ids, labels=ids)
            ce_clean += loss / n_local
            local = cap.stacked()
            x = gather(local).float() * shard.norm_factor[None, :, None]
            partial, codes = shard.decode_partial(x)
            dist.all_reduce(partial)
            n_tokens += x.shape[0]
            se += (partial - x).pow(2).sum(-1).sum(0)
            sx += x.pow(2).sum(-1).sum(0)
            for i, (_, a) in codes.items():
                l0[i] += (a > 0).float().sum(-1).mean() / n_local
                fire[i] += (a > 0).sum(0).double()
            # (b*T, L, D) this rank's own tokens, model scale
            mine = (partial[rank * local.shape[0]:][:local.shape[0]] / shard.norm_factor[None, :, None]).to(local.dtype)
            for j in range(N_LAYERS):
                h = layers[j].register_forward_hook(lambda m, i, o, j=j: mine[:, j].reshape_as(o))
                _, loss, _ = lm(input_ids=ids, labels=ids)
                h.remove()
                ce_sae[j] += loss / n_local
    for t in (ce_sae, ce_clean):
        dist.all_reduce(t)
        t /= world
    # each encoder layer lives on one rank
    dist.all_reduce(l0)
    # se / sx are already global (every rank saw every gathered token)
    recon = (se / sx * 100).tolist()
    stats = {}  # {"L<j>": {...}} this rank's encoder layers' feature stats
    (results_dir / "density").mkdir(exist_ok=True)
    for i in shard.layers:
        # (F,) float64 firing fraction on eval tokens
        d = (fire[i] / n_tokens).cpu().numpy()
        d.tofile(results_dir / "density" / f"L{i}.density.bin")
        stats[f"L{i}"] = {"dead_frac_train": (getattr(shard, f"tokens_since_fired_{i}") > 10_000_000).float().mean().item(),
                          "density_hist": [float((d == 0).mean())] + (np.histogram(np.log10(d[d > 0]), bins=[-np.inf, -5, -4, -3, -2, -1, 0.0001])[0] / len(d)).tolist()}
    all_stats = [None] * world
    dist.all_gather_object(all_stats, stats)
    if rank == 0:
        saes = {}  # {"L<j>": {recon_err_pct, ce_increase_pct, l0, dead_frac_train, density_hist}}
        for j in range(N_LAYERS):
            saes[f"L{j}"] = {"recon_err_pct": recon[j], "ce_increase_pct": ((ce_sae[j] - ce_clean) / ce_clean * 100).item(),
                             "l0": l0[j].item(), **next(s[f"L{j}"] for s in all_stats if f"L{j}" in s)}
        core = {"ce_clean": ce_clean.item(), "n_eval_tokens": n_tokens, "l0_total": l0.sum().item(),
                "density_bins": ["never", "<1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", "1e-2..1e-1", ">=1e-1"],
                "config": {"features_per_layer": n_features, "lam": lam, "train_tokens": max_steps * step_chunks * T, "lr": LR,
                           "lr_decay_frac": LR_DECAY_FRAC, "eval_tokens": n_local * world * eval_chunks * T},
                "saes": saes}
        (results_dir / "core.json").write_text(json.dumps(core, indent=2))
        print(json.dumps(core, indent=2), flush=True)
        print(f"wcc done: weights in {weights_dir}, core.json + density in {results_dir}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]))
