import json
import torch
import torch.nn as nn
from pathlib import Path

from evoke.OlMo2_1b.run.loader import build_model_and_tokenizer
from synapse.train.simple_train import simple_train, simple_eval
from synapse.train.data_to_loaders import BinUnsupervisedDataset
from synapse.probes.sae.JumpReluSAE import JumpReLU_SAE
from synapse.probes.sae.SlabConeSAE import SlabConeSAE
from synapse.probes.sae.TopKSAE import TopKSAE


BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "OlMo2_1b" / "sae_compare"
D_IN = 2048
EXPANSION = 16  # d_sae = 2048 * 16 = 32768
INIT_COEFF = 0.1
TOPK_KS = [64, 96, 128]
MAX_STEPS = 10_000
BATCH_SIZE = 4
LAYER_IDX = 8


class ActivationTrainer(nn.Module):
    def __init__(self, lm, sae, layer_idx=LAYER_IDX):
        super().__init__()
        self.lm = lm
        self.sae = sae
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad = False
        self._acts = None
        self._hook_handle = self.lm.model.layers[layer_idx].register_forward_hook(self._capture)

    def _capture(self, module, input, output):
        self._acts = output[0].detach()  # (B, seq, 2048)

    def compute_loss(self, token_chunk):
        with torch.no_grad():
            self.lm.model(input_ids=token_chunk)
        acts = self._acts.reshape(-1, self._acts.shape[-1])  # (B*seq, 2048)
        return self.sae.compute_loss(acts)

    def remove_hook(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def train(self, mode=True):
        super().train(mode)
        self.lm.eval()
        return self


def _run(name, sae, results):
    trainer = ActivationTrainer(lm, sae)
    opt = torch.optim.Adam(trainer.parameters(), lr=7e-5, betas=(0.0, 0.999))

    save_dir = WEIGHTS_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)

    simple_train(
        model=trainer,
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        optimizer=opt,
        epochs=1,
        save_path=str(save_dir),
        batches_per_log=100,
        batches_per_save=100,
        eval_dataset=eval_ds,
        eval_batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
    )

    print(f"\n--- {name} final eval ---")
    eval_metrics = simple_eval(trainer, eval_ds, BATCH_SIZE, batches=10000)
    trainer.remove_hook()
    results[name] = eval_metrics
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")


def main():
    global lm, train_ds, eval_ds

    meta = json.loads((BIN_DIR / "meta.json").read_text())
    train_ds = BinUnsupervisedDataset(str(BIN_DIR / "train.bin"), torch.int32, tuple(meta["train_shape"]))
    eval_ds = BinUnsupervisedDataset(str(BIN_DIR / "eval.bin"), torch.int32, tuple(meta["eval_shape"]))
    lm, _, _ = build_model_and_tokenizer()

    results = {}

    # JumpReLU — adaptive sparsity, matches SlabCone mechanism
    print(f"\n{'='*60}")
    print(f"=== jumprelu (coeff={INIT_COEFF}) ===")
    print(f"{'='*60}")
    _run("jumprelu",
         JumpReLU_SAE(D_IN, EXPANSION, sparsity_coeff=INIT_COEFF),
         results)

    # SlabCone — original gates, same adaptive sparsity mechanism
    print(f"\n{'='*60}")
    print(f"=== slabcone (coeff={INIT_COEFF}) ===")
    print(f"{'='*60}")
    _run("slabcone",
         SlabConeSAE(D_IN, EXPANSION, sparsity_coeff=INIT_COEFF),
         results)

    # TopK — fixed k sweep, no sparsity penalty
    for k in TOPK_KS:
        print(f"\n{'='*60}")
        print(f"=== topk (k={k}) ===")
        print(f"{'='*60}")
        _run(f"topk/k_{k}",
             TopKSAE(D_IN, EXPANSION, k=k),
             results)

    print(f"\n{'='*60}")
    print("=== COMPARISON ===")
    print(f"{'='*60}")
    header = f"{'SAE':<16} {'recon_pct':>10} {'active':>10}"
    print(header)
    print("-" * len(header))
    for name in ["jumprelu", "slabcone"] + [f"topk/k_{k}" for k in TOPK_KS]:
        label = name.replace("topk/", "topk ")
        m = results.get(name, {})
        recon = m.get("recon_pct", 0)
        active = m.get("active", m.get("l0", 0))
        print(f"{label:<16} {recon:>10.1f} {active:>10.1f}")


if __name__ == "__main__":
    main()
