# TopK SAEs (k=64, 16x) on OlMo2-1B resid_post at all 16 layers, then core bench per layer, picks for every
# feature, and labels + detection scores for every feature. every phase resumes from what is already on disk.
# run from repo root: python -m evoke.OlMo2_1b.interp.sae_resid_topk
# outputs: weights/evoke/OlMo2_1b/sae_resid_topk/L<i>.pt
#          results/OlMo2_1b/sae_resid_topk/{core.json, picks/, autointerp.json, summary.md}

import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from evoke.OlMo2_1b.run.loader import build_model_and_tokenizer
from evoke.OlMo2_1b.interp.hooked_olmo_sae import HookedOlmoSAE
from evoke.OlMo2_1b.interp.sae_bench_compare import evaluate_core
from synapse.interp.gather import gather_picks
from synapse.interp.label import label_units
from synapse.probes.multi_sae_trainer import MultiSAETrainer
from synapse.probes.sae.BatchTopKSAE import BatchTopKSAE
from synapse.train.data_to_loaders import BinUnsupervisedDataset, dataset_to_dataloader
from synapse.train.simple_train import simple_train, _to_cuda

BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "OlMo2_1b" / "sae_resid_topk"
RESULTS_DIR = Path.cwd() / "results" / "OlMo2_1b" / "sae_resid_topk"
LAYERS = list(range(16))
GROUP_SIZE = 4  # SAEs trained together in one LM pass; 16 at once does not fit 32GB with adam state
D_IN = 2048
EXPANSION = 16
K = 64
TRAIN_TOKENS = 100_000_000
BATCH_CHUNKS = 64  # x 128 = 8192 tokens per step
LR = 3e-4
EVAL_BATCHES = 50
PICK_CHUNKS = 16000  # x 128 = 2.05M tokens
PICK_BATCH_CHUNKS = 8
MODEL = "deepseek/deepseek-v4-flash-0731"
WORKERS = 300


def make_sae(expansion):
    return BatchTopKSAE(D_IN, expansion, k=K, per_token=True)


def train_group(lm, layers, expansion, train_ds, eval_ds, train_tokens, batch_chunks, chunk_size, weights_dir, results_dir):
    saes = {i: make_sae(expansion) for i in layers}
    trainer = MultiSAETrainer(lm.model, saes)
    # per layer input scaling so the mean residual norm is sqrt(d_in), estimated on a few batches
    dl = iter(dataset_to_dataloader(train_ds, batch_chunks, cur_epoch=1))
    norms = {i: 0.0 for i in layers}  # {layer: mean resid norm}
    with torch.no_grad():
        for _ in range(4):
            trainer.lm(input_ids=_to_cuda(next(dl)))
            for i in layers:
                norms[i] += trainer._acts[i].float().norm(dim=-1).mean().item() / 4
    for i in layers:
        saes[i].norm_factor.fill_(D_IN ** 0.5 / norms[i])
    print("mean resid norms:", {i: round(v, 2) for i, v in norms.items()}, flush=True)

    # checkpoints go to a per-group dir: simple_train deletes other .pt files in its save dir
    group_dir = weights_dir / f"group_L{layers[0]}-L{layers[-1]}"
    group_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(trainer.parameters(), lr=LR)
    simple_train(trainer, train_ds, batch_chunks, opt, epochs=1, save_path=str(group_dir), batches_per_log=200,
                 batches_per_save=2000, eval_dataset=eval_ds, max_steps=train_tokens // (batch_chunks * chunk_size))
    trainer.remove_hooks()
    for i in layers:
        torch.save(saes[i].state_dict(), weights_dir / f"L{i}.pt")
    for f in ("losses.json", "losses.png"):
        if (group_dir / f).exists():
            shutil.copy(group_dir / f, results_dir / f"{group_dir.name}.{f}")
    return saes


def main(layers=LAYERS, group_size=GROUP_SIZE, expansion=EXPANSION, train_tokens=TRAIN_TOKENS, batch_chunks=BATCH_CHUNKS,
         eval_batches=EVAL_BATCHES, pick_chunks=PICK_CHUNKS, label_limit=None, weights_dir=WEIGHTS_DIR, results_dir=RESULTS_DIR):
    # label_limit: label only the first n units per layer (None = every unit); for smoke tests
    torch.backends.cuda.matmul.allow_tf32 = True  # tf32 matmuls: ~2x faster SAE training, standard for SAE training
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((BIN_DIR / "meta.json").read_text())
    train_ds = BinUnsupervisedDataset(str(BIN_DIR / "train.bin"), torch.int32, tuple(meta["train_shape"]))
    eval_ds = BinUnsupervisedDataset(str(BIN_DIR / "eval.bin"), torch.int32, tuple(meta["eval_shape"]))
    lm, tokenizer, _ = build_model_and_tokenizer()
    names = {i: f"L{i}" for i in layers}  # {layer: sae / hook name}

    # --- phase 1: train + core eval, group by group (skips groups whose weights exist) ---
    core_path = results_dir / "core.json"
    core = json.loads(core_path.read_text()) if core_path.exists() else {"saes": {}}
    for g in range(0, len(layers), group_size):
        group = layers[g:g + group_size]
        if all((weights_dir / f"L{i}.pt").exists() for i in group) and all(names[i] in core["saes"] for i in group):
            print(f"group {group}: already trained + evaluated, skipping", flush=True)
            continue
        print(f"=== training layers {group} ===", flush=True)
        saes = train_group(lm, group, expansion, train_ds, eval_ds, train_tokens, batch_chunks, meta["chunk_size"], weights_dir, results_dir)
        c = evaluate_core(lm, {names[i]: saes[i] for i in group}, {names[i]: i for i in group}, eval_ds, eval_batches, batch_chunks)
        core["ce_clean"] = c["ce_clean"]
        core["saes"].update(c["saes"])
        core["config"] = {"k": K, "expansion": expansion, "train_tokens": train_tokens, "lr": LR, "group_size": group_size,
                          "eval_tokens": eval_batches * batch_chunks * meta["chunk_size"]}
        core_path.write_text(json.dumps(core, indent=2))
        print(json.dumps({n: core["saes"][n] for n in (names[i] for i in group)}, indent=2), flush=True)
        del saes
        torch.cuda.empty_cache()

    # --- phase 2: picks for every feature of every layer (one pass, all SAEs loaded) ---
    picks_dir = results_dir / "picks"
    saes = {}  # {name: (layer, sae)}
    for i in layers:
        sae = make_sae(expansion)
        sae.load_state_dict(torch.load(weights_dir / f"L{i}.pt"))
        saes[names[i]] = (i, sae.cuda().eval())
    if not (picks_dir / "meta.json").exists():
        gather_picks(HookedOlmoSAE(lm, saes), BIN_DIR / "train.bin", tuple(meta["train_shape"]), list(saes), pick_chunks, picks_dir, PICK_BATCH_CHUNKS)
    del saes
    torch.cuda.empty_cache()

    # --- phase 3: label + score every feature (resumes from the jsonl files) ---
    units = {names[i]: list(range(label_limit)) for i in layers} if label_limit else None
    asyncio.run(label_units(picks_dir, tokenizer, MODEL, [names[i] for i in layers], workers=WORKERS, units=units))

    # --- summary ---
    pm = json.loads((picks_dir / "meta.json").read_text())
    K_picks = pm["top_k"] + pm["iw_k"]
    summary = {}
    rows = []
    for i in layers:
        n = names[i]
        chunk = np.fromfile(picks_dir / f"{n}.pick_chunk.bin", dtype=np.int32).reshape(pm["hooks"][n], K_picks)
        rs = [json.loads(l) for l in (picks_dir / f"{n}.labels.jsonl").read_text().splitlines()]
        sc = np.array([r["score"] for r in rs if r["score"] is not None])
        cs = core["saes"][n]
        summary[n] = {**cs, "never_fired_frac": float((chunk[:, 0] < 0).mean()), "full_picks_frac": float((chunk >= 0).all(1).mean()),
                      "n_scored": int(len(sc)), "autointerp_mean": float(sc.mean()), "autointerp_median": float(np.median(sc)),
                      "autointerp_frac_ge_0.7": float((sc >= 0.7).mean())}
        s = summary[n]
        rows.append(f"| L{i} | {s['l0']:.1f} | {s['expl_var']:.3f} | {s['ce_recovered']:.3f} | {s['never_fired_frac']:.1%} | "
                    f"{s['full_picks_frac']:.0%} | {s['autointerp_mean']:.3f} | {s['autointerp_median']:.2f} | {s['autointerp_frac_ge_0.7']:.0%} |")
    (results_dir / "autointerp.json").write_text(json.dumps(summary, indent=2))
    table = "\n".join(["| layer | L0 | explained var | CE recovered | never fired | >=40 firings | autointerp mean | median | >=0.7 |",
                       "|---|---|---|---|---|---|---|---|---|", *rows])
    (results_dir / "summary.md").write_text(
        f"# TopK SAEs (k={K}, {expansion}x) on OlMo2-1B resid_post, all layers\n\n"
        f"train {train_tokens:,} tokens per SAE, lr {LR}, groups of {group_size}. picks over {pick_chunks * meta['chunk_size']:,} tokens. "
        f"labeler {MODEL}. clean CE {core['ce_clean']:.3f}.\n\n{table}\n")
    print(table, flush=True)


if __name__ == "__main__":
    main()
