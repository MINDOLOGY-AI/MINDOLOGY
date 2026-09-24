# trains TopK / BatchTopK / Matryoshka-BatchTopK SAEs (k=64, 16x) on OlMo2-1B resid_post.8 in one LM pass,
# then runs the core bench (L0, explained variance, CE recovered with the SAE spliced into the model).
# run from repo root: python -m evoke.OlMo2_1b.interp.sae_bench_compare
# outputs: weights/evoke/OlMo2_1b/sae_bench_compare/<name>.pt, results/OlMo2_1b/sae_bench_compare/core.json

import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from evoke.OlMo2_1b.run.loader import build_model_and_tokenizer
from synapse.algorithms.encoding.mask import create_olmo_causal_mask
from synapse.probes.sae.BatchTopKSAE import BatchTopKSAE
from synapse.train.data_to_loaders import BinUnsupervisedDataset, dataset_to_dataloader
from synapse.train.simple_train import simple_train, _to_cuda

BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "OlMo2_1b" / "sae_bench_compare"
RESULTS_DIR = Path.cwd() / "results" / "OlMo2_1b" / "sae_bench_compare"
LAYER = 8  # resid_post.8: residual stream after layer 8 of 16
D_IN = 2048
EXPANSION = 16
K = 64
TRAIN_TOKENS = 100_000_000
BATCH_CHUNKS = 64  # x 128 = 8192 tokens per step
LR = 3e-4
MATRYOSHKA = [1 / 32, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2]
EVAL_BATCHES = 100


def make_saes(expansion):
    return {
        "topk": BatchTopKSAE(D_IN, expansion, k=K, per_token=True),
        "batchtopk": BatchTopKSAE(D_IN, expansion, k=K),
        "matryoshka": BatchTopKSAE(D_IN, expansion, k=K, matryoshka_fractions=MATRYOSHKA),
    }


class SameLayerSAETrainer(nn.Module):
    # several SAEs on the same residual layer, one truncated LM forward per step (layers past LAYER are skipped)
    def __init__(self, lm, saes):
        super().__init__()
        self._lm = [lm]  # list wrapper keeps the frozen LM out of the module registry / checkpoints
        lm.eval()
        for p in lm.parameters():
            p.requires_grad = False
        self.saes = nn.ModuleDict(saes)

    def resid(self, ids):
        # ids: (b, L) -> (b*L, d_in) residual stream after LAYER, fp32
        m = self._lm[0].model
        h = m.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        pe = m.rotary_emb(h, pos)
        mask = create_olmo_causal_mask(h)
        for layer in m.layers[:LAYER + 1]:
            h, _ = layer(h, mask, pe)
        return h.reshape(-1, h.shape[-1]).float()

    def compute_loss(self, token_chunk):
        with torch.no_grad():
            acts = self.resid(token_chunk)
        total = 0.0
        metrics = {}
        for name, sae in self.saes.items():
            loss, m = sae.compute_loss(acts)
            total = total + loss
            metrics.update({f"{name}_{k}": v for k, v in m.items()})
        return total, metrics

    def train(self, mode=True):
        super().train(mode)
        self._lm[0].eval()
        return self


def evaluate_core(lm, saes, layers, eval_ds, n_batches, batch_chunks):
    # per sae: CE with its layer's output replaced by the SAE reconstruction, and by zeros (per layer), plus recon stats
    # saes: {name: sae}, layers: {name: layer index the sae reads}
    dl = dataset_to_dataloader(eval_ds, batch_chunks, cur_epoch=0)
    stats = {name: {"ce": 0.0, "ce_zero": 0.0, "mse": 0.0, "expl_var": 0.0, "l0": 0.0} for name in saes}
    # {name: (D,) float64} how many eval tokens each feature fired on
    fire_counts = {name: torch.zeros(sae.d_sae, dtype=torch.float64, device="cuda") for name, sae in saes.items()}
    n_tokens = 0
    ce_clean = 0.0
    replacement = {}  # {"fn": callable resid -> resid} set per pass

    def hook(module, inputs, output):
        return (replacement["fn"](output[0]), *output[1:])

    def ce_with(layer_idx, fn, ids, labels):
        replacement["fn"] = fn
        h = lm.model.layers[layer_idx].register_forward_hook(hook)
        _, loss, _ = lm(input_ids=ids, labels=labels)
        h.remove()
        return loss.item()

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
                st = stats[name]

                def splice(resid, sae=sae, st=st, fc=fire_counts[name]):
                    b, L, d = resid.shape
                    x = resid.reshape(-1, d).float()
                    f = sae.encode(x)
                    x_hat = sae.decode(f)
                    st["mse"] += (x_hat - x).pow(2).sum(-1).mean().item() / n_batches
                    st["expl_var"] += (1 - (x_hat - x).pow(2).sum(-1).mean() / x.var(dim=0).sum()).item() / n_batches
                    st["l0"] += (f > 0).float().sum(-1).mean().item() / n_batches
                    fc += (f > 0).sum(0).double()
                    return x_hat.to(resid.dtype).view(b, L, d)
                st["ce"] += ce_with(layers[name], splice, ids, labels) / n_batches
                st["ce_zero"] += ce_with(layers[name], torch.zeros_like, ids, labels) / n_batches
    # {name: (D,) float64} fraction of eval tokens each feature fires on
    density = {name: (c / n_tokens).cpu().numpy() for name, c in fire_counts.items()}
    for name, st in stats.items():
        st["ce_recovered"] = (st["ce_zero"] - st["ce"]) / (st["ce_zero"] - ce_clean)
        st["dead_frac_train"] = (saes[name].tokens_since_fired > saes[name].dead_tokens).float().mean().item()
        d = density[name]
        # share of features per log10 density bin: never fired on eval, <1e-5, 1e-5..1e-4, ..., >=1e-1
        st["density_hist"] = [float((d == 0).mean())] + (np.histogram(np.log10(d[d > 0]), bins=[-np.inf, -5, -4, -3, -2, -1, 0.0001])[0] / len(d)).tolist()
    return {"ce_clean": ce_clean, "n_eval_tokens": n_tokens, "density_bins": ["never", "<1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", "1e-2..1e-1", ">=1e-1"], "saes": stats}, density


def main(train_tokens=TRAIN_TOKENS, expansion=EXPANSION, batch_chunks=BATCH_CHUNKS, eval_batches=EVAL_BATCHES,
         weights_dir=WEIGHTS_DIR, results_dir=RESULTS_DIR):
    meta = json.loads((BIN_DIR / "meta.json").read_text())
    train_ds = BinUnsupervisedDataset(str(BIN_DIR / "train.bin"), torch.int32, tuple(meta["train_shape"]))
    eval_ds = BinUnsupervisedDataset(str(BIN_DIR / "eval.bin"), torch.int32, tuple(meta["eval_shape"]))
    lm, _, _ = build_model_and_tokenizer()
    saes = make_saes(expansion)
    trainer = SameLayerSAETrainer(lm, saes).cuda()

    # scale inputs so the mean residual norm is sqrt(d_in), estimated on a few batches
    dl = iter(dataset_to_dataloader(train_ds, batch_chunks, cur_epoch=1))
    with torch.no_grad():
        mean_norm = sum(trainer.resid(_to_cuda(next(dl))).norm(dim=-1).mean().item() for _ in range(4)) / 4
    for sae in saes.values():
        sae.norm_factor.fill_(D_IN ** 0.5 / mean_norm)
    print(f"mean resid norm {mean_norm:.2f} -> norm_factor {D_IN ** 0.5 / mean_norm:.4f}")

    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    max_steps = train_tokens // (batch_chunks * meta["chunk_size"])
    opt = torch.optim.Adam(trainer.parameters(), lr=LR, betas=(0.9, 0.999))
    simple_train(trainer, train_ds, batch_chunks, opt, epochs=1, save_path=str(weights_dir),
                 batches_per_log=200, batches_per_save=2000, eval_dataset=eval_ds, max_steps=max_steps)
    for name, sae in saes.items():
        torch.save(sae.state_dict(), weights_dir / f"{name}.pt")
    for f in ("losses.json", "losses.png"):
        if (weights_dir / f).exists():
            shutil.copy(weights_dir / f, results_dir / f)

    core, _ = evaluate_core(lm, saes, {name: LAYER for name in saes}, eval_ds, eval_batches, batch_chunks)
    core["config"] = {"layer": LAYER, "k": K, "expansion": expansion, "train_tokens": max_steps * batch_chunks * meta["chunk_size"],
                      "matryoshka_fractions": MATRYOSHKA, "lr": LR, "eval_tokens": eval_batches * batch_chunks * meta["chunk_size"]}
    (results_dir / "core.json").write_text(json.dumps(core, indent=2))
    print(json.dumps(core, indent=2))


if __name__ == "__main__":
    main()
