# TopK SAEs (k=64, 16x) on OlMo2-1B resid_post, 1B training tokens each, then core bench per layer, picks for every
# feature, and labels + detection scores for every feature. every phase resumes from what is already on disk.
# run from repo root: python -m evoke.OlMo2_1b.interp.sae_resid_topk
# outputs: weights/evoke/OlMo2_1b/sae_resid_topk_1bTok/L<i>.pt
#          results/OlMo2_1b/sae_resid_topk_1bTok/{core.json, density/, picks/, labels/, autointerp.json, summary.md}

import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from evoke.OlMo2_1b.run.loader import load_olmo2_model, load_olmo2_tokenizer
from evoke.OlMo2_1b.interp.hooked_olmo_sae import HookedOlmoSAE
from synapse.interp.picks import gather_picks
from synapse.interp.label import label_units
from synapse.probes.multi_sae_trainer import MultiSAETrainer
from synapse.probes.sae.TopKSAE import TopKSAE
from synapse.train.data_to_loaders import BinUnsupervisedDataset, dataset_to_dataloader
from synapse.train.simple_train import simple_train, _to_cuda

BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "OlMo2_1b" / "sae_resid_topk_1bTok"
RESULTS_DIR = Path.cwd() / "results" / "OlMo2_1b" / "sae_resid_topk_1bTok"
LAYERS = list(range(16))
GROUP_SIZE = 4  # SAEs trained together in one LM pass; 16 at once does not fit 32GB with adam state
D_IN = 2048
EXPANSION = 16
K = 64
TRAIN_TOKENS = 1_000_000_000
BATCH_CHUNKS = 64  # x 128 = 8192 tokens per step
LR = 3e-4
LR_DECAY_FRAC = 0.2  # lr constant, then linear to 0 over this final fraction of steps (dictionary_learning TopK recipe)
EVAL_BATCHES = 50
PICK_CHUNKS = 16000  # x 128 = 2.05M tokens
PICK_BATCH_CHUNKS = 8
MODEL = "deepseek/deepseek-v4-flash-0731"
WORKERS = 200
SEED = 21  # global torch seed: SAE init and the picks' importance-weighted draws


def make_sae(expansion):
    return TopKSAE(D_IN, expansion, k=K)


def train_group(lm, layers, expansion, train_ds, eval_ds, train_tokens, batch_chunks, chunk_size, weights_dir, results_dir):
    saes = {i: make_sae(expansion) for i in layers}
    trainer = MultiSAETrainer(lm.model, saes)
    # per layer input scaling so the mean residual norm is sqrt(d_in), estimated on a few batches
    dl = iter(dataset_to_dataloader(train_ds, batch_chunks, cur_epoch=1))
    norms = {i: 0.0 for i in layers}  # {layer: mean resid norm}
    with torch.no_grad():
        for _ in range(4):
            trainer.run(_to_cuda(next(dl)))
            for i in layers:
                norms[i] += trainer._acts[i].float().norm(dim=-1).mean().item() / 4
    for i in layers:
        saes[i].norm_factor.fill_(D_IN ** 0.5 / norms[i])
    print("mean resid norms:", {i: round(v, 2) for i, v in norms.items()}, flush=True)

    # checkpoints go to a per-group dir: simple_train deletes other .pt files in its save dir
    group_dir = weights_dir / f"group_L{layers[0]}-L{layers[-1]}"
    group_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(trainer.parameters(), lr=LR)
    max_steps = train_tokens // (batch_chunks * chunk_size)
    decay_start = int(max_steps * (1 - LR_DECAY_FRAC))
    # lr multiplier per step: 1 until decay_start, then linear down to 0 at max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0, (max_steps - step) / (max_steps - decay_start)))
    simple_train(trainer, train_ds, batch_chunks, opt, epochs=1, save_path=str(group_dir), batches_per_log=200,
                 batches_per_save=2000, eval_dataset=eval_ds, max_steps=max_steps, scheduler=sched)
    trainer.remove_hooks()
    for i in layers:
        torch.save(saes[i].state_dict(), weights_dir / f"L{i}.pt")
    for f in ("losses.json", "losses.png"):
        if (group_dir / f).exists():
            shutil.copy(group_dir / f, results_dir / f"{group_dir.name}.{f}")
    return saes


def evaluate_core(lm, saes, layers, eval_ds, n_batches, batch_chunks):
    # per sae: recon_err_pct = sum ||x - x_hat||^2 / sum ||x||^2 * 100 over all eval tokens,
    # ce_increase_pct = (CE with the layer's output replaced by x_hat - clean CE) / clean CE * 100, plus l0 and density
    # saes: {name: sae}, layers: {name: layer index the sae reads}
    dl = dataset_to_dataloader(eval_ds, batch_chunks, cur_epoch=0)
    stats = {name: {"ce_sae": 0.0, "l0": 0.0} for name in saes}
    # {name: [sum ||x - x_hat||^2, sum ||x||^2]} over all eval tokens
    sq = {name: [0.0, 0.0] for name in saes}
    # {name: (D,) float64} how many eval tokens each feature fired on
    fire_counts = {name: torch.zeros(sae.d_sae, dtype=torch.float64, device="cuda") for name, sae in saes.items()}
    n_tokens = 0
    ce_clean = 0.0
    for sae in saes.values():
        sae.eval()
    with torch.no_grad():
        for i, ids in enumerate(dl):
            if i >= n_batches:
                break
            ids = _to_cuda(ids)
            labels = ids.long()
            _, loss, _ = lm(input_ids=ids, labels=labels)
            ce_clean += loss.item() / n_batches
            n_tokens += ids.numel()
            for name, sae in saes.items():
                st, fc = stats[name], fire_counts[name]

                # replaces the residual after the sae's layer with its reconstruction for this forward pass
                def splice(module, inputs, output, sae=sae, st=st, fc=fc, sq=sq[name]):
                    resid = output[0]
                    b, L, d = resid.shape
                    # (b, L, d) -> (b*L, d)
                    x = resid.reshape(-1, d).float()
                    # (b*L, d) -> (b*L, d_sae)
                    f = sae.encode(x)
                    # (b*L, d_sae) -> (b*L, d)
                    x_hat = sae.decode(f)
                    sq[0] += (x_hat - x).pow(2).sum().item()
                    sq[1] += x.pow(2).sum().item()
                    st["l0"] += (f > 0).float().sum(-1).mean().item() / n_batches
                    fc += (f > 0).sum(0).double()
                    return (x_hat.to(resid.dtype).view(b, L, d), *output[1:])
                h = lm.model.layers[layers[name]].register_forward_hook(splice)
                _, loss, _ = lm(input_ids=ids, labels=labels)
                h.remove()
                st["ce_sae"] += loss.item() / n_batches
    # {name: (D,) float64} fraction of eval tokens each feature fires on
    density = {name: (c / n_tokens).cpu().numpy() for name, c in fire_counts.items()}
    for name, st in stats.items():
        st["recon_err_pct"] = sq[name][0] / sq[name][1] * 100
        st["ce_increase_pct"] = (st["ce_sae"] - ce_clean) / ce_clean * 100
        st["dead_frac_train"] = (saes[name].tokens_since_fired > saes[name].dead_tokens).float().mean().item()
        d = density[name]
        # share of features per log10 density bin: never fired on eval, <1e-5, 1e-5..1e-4, ..., >=1e-1
        st["density_hist"] = [float((d == 0).mean())] + (np.histogram(np.log10(d[d > 0]), bins=[-np.inf, -5, -4, -3, -2, -1, 0.0001])[0] / len(d)).tolist()
    return {"ce_clean": ce_clean, "n_eval_tokens": n_tokens, "density_bins": ["never", "<1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", "1e-2..1e-1", ">=1e-1"], "saes": stats}, density


def main(layers=LAYERS, group_size=GROUP_SIZE, expansion=EXPANSION, train_tokens=TRAIN_TOKENS, batch_chunks=BATCH_CHUNKS,
         eval_batches=EVAL_BATCHES, pick_chunks=PICK_CHUNKS, label_limit=None, weights_dir=WEIGHTS_DIR, results_dir=RESULTS_DIR):
    # label_limit: label only the first n units per layer (None = every unit); for smoke tests
    torch.backends.cuda.matmul.allow_tf32 = True  # tf32 matmuls: ~2x faster SAE training, standard for SAE training
    torch.manual_seed(SEED)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((BIN_DIR / "meta.json").read_text())
    train_ds = BinUnsupervisedDataset(str(BIN_DIR / "train.bin"), torch.int32, tuple(meta["train_shape"]))
    eval_ds = BinUnsupervisedDataset(str(BIN_DIR / "eval.bin"), torch.int32, tuple(meta["eval_shape"]))
    names = {i: f"L{i}" for i in layers}  # {layer: sae / hook name}
    core_path = results_dir / "core.json"
    core = json.loads(core_path.read_text()) if core_path.exists() else {"saes": {}}
    picks_dir = results_dir / "picks"
    labels_dir = results_dir / "labels"
    # the LM is only needed for train / eval / picks; labeling alone needs just the tokenizer
    need_lm = not (picks_dir / "meta.json").exists() or not all("recon_err_pct" in core["saes"].get(names[i], {}) for i in layers)
    lm = load_olmo2_model()[0] if need_lm else None
    tokenizer = load_olmo2_tokenizer()

    # --- phase 1: train + core eval, group by group (skips groups whose weights exist) ---
    for g in range(0, len(layers), group_size):
        group = layers[g:g + group_size]
        if all("recon_err_pct" in core["saes"].get(names[i], {}) for i in group):
            print(f"group {group}: already trained + evaluated, skipping", flush=True)
            continue
        if all((weights_dir / f"L{i}.pt").exists() for i in group):
            print(f"=== layers {group}: weights exist, loading for eval ===", flush=True)
            saes = {i: make_sae(expansion) for i in group}
            for i in group:
                saes[i].load_state_dict(torch.load(weights_dir / f"L{i}.pt"))
                saes[i].cuda()
        else:
            print(f"=== training layers {group} ===", flush=True)
            saes = train_group(lm, group, expansion, train_ds, eval_ds, train_tokens, batch_chunks, meta["chunk_size"], weights_dir, results_dir)
        c, density = evaluate_core(lm, {names[i]: saes[i] for i in group}, {names[i]: i for i in group}, eval_ds, eval_batches, batch_chunks)
        (results_dir / "density").mkdir(exist_ok=True)
        for n, d in density.items():
            d.tofile(results_dir / "density" / f"{n}.density.bin")  # (D,) float64 firing fraction on eval tokens
        core["ce_clean"] = c["ce_clean"]
        core["density_bins"] = c["density_bins"]
        core["saes"].update(c["saes"])
        core["config"] = {"k": K, "expansion": expansion, "train_tokens": train_tokens, "lr": LR, "lr_decay_frac": LR_DECAY_FRAC, "group_size": group_size,
                          "eval_tokens": eval_batches * batch_chunks * meta["chunk_size"]}
        core_path.write_text(json.dumps(core, indent=2))
        print(json.dumps({n: core["saes"][n] for n in (names[i] for i in group)}, indent=2), flush=True)
        del saes
        torch.cuda.empty_cache()

    # --- phase 2: picks for every feature of every layer (one pass, all SAEs loaded) ---
    if not (picks_dir / "meta.json").exists():
        saes = {}  # {name: (layer, sae)}
        for i in layers:
            sae = make_sae(expansion)
            sae.load_state_dict(torch.load(weights_dir / f"L{i}.pt"))
            saes[names[i]] = (i, sae.cuda().eval())
        gather_picks(HookedOlmoSAE(lm, saes), BIN_DIR / "train.bin", tuple(meta["train_shape"]), list(saes), pick_chunks, picks_dir, PICK_BATCH_CHUNKS)
        del saes
        torch.cuda.empty_cache()

    # --- phase 3: label + score every feature (resumes from the jsonl files) ---
    units = {names[i]: list(range(label_limit)) for i in layers} if label_limit else None
    n_failed = asyncio.run(label_units(picks_dir, labels_dir, tokenizer, MODEL, [names[i] for i in layers], workers=WORKERS,
                                       units=units, unit_word="feature"))
    assert n_failed == 0, f"{n_failed} units failed labeling, see {labels_dir / 'errors.json'}; rerun to retry them"

    # --- summary ---
    pm = json.loads((picks_dir / "meta.json").read_text())
    K_picks = pm["top_k"] + pm["iw_k"]
    summary = {}
    rows = []
    for i in layers:
        n = names[i]
        chunk = np.fromfile(picks_dir / f"{n}.pick_chunk.bin", dtype=np.int32).reshape(pm["hooks"][n], K_picks)
        rs = [json.loads(l) for l in (labels_dir / f"{n}.jsonl").read_text().splitlines()]
        sc = np.array([r["score"] for r in rs if r["score"] is not None])
        cs = core["saes"][n]
        summary[n] = {**cs, "never_fired_frac": float((chunk[:, 0] < 0).mean()), "full_picks_frac": float((chunk >= 0).all(1).mean()),
                      "n_scored": int(len(sc)), "autointerp_mean": float(sc.mean()), "autointerp_median": float(np.median(sc)),
                      "autointerp_frac_ge_0.7": float((sc >= 0.7).mean())}
        s = summary[n]
        rows.append(f"| L{i} | {s['l0']:.1f} | {s['recon_err_pct']:.1f}% | {s['ce_increase_pct']:.1f}% | {s['never_fired_frac']:.1%} | "
                    f"{s['full_picks_frac']:.0%} | {s['autointerp_mean']:.3f} | {s['autointerp_median']:.2f} | {s['autointerp_frac_ge_0.7']:.0%} |")
    (results_dir / "autointerp.json").write_text(json.dumps(summary, indent=2))
    table = "\n".join(["| layer | L0 | recon err | CE increase | never fired | >=40 firings | autointerp mean | median | >=0.7 |",
                       "|---|---|---|---|---|---|---|---|---|", *rows])
    (results_dir / "summary.md").write_text(
        f"# TopK SAEs (k={K}, {expansion}x) on OlMo2-1B resid_post, all layers\n\n"
        f"train {train_tokens:,} tokens per SAE, lr {LR} (linear decay over the last {LR_DECAY_FRAC:.0%}), groups of {group_size}. picks over {pick_chunks * meta['chunk_size']:,} tokens. "
        f"labeler {MODEL}. clean CE {core['ce_clean']:.3f}.\n\n{table}\n")
    print(table, flush=True)


if __name__ == "__main__":
    main()
