import json
import torch
import torch.nn as nn
from pathlib import Path

from evoke.OlMo2_1b.run.loader import build_model_and_tokenizer
from synapse.train.simple_train import simple_train, simple_eval
from synapse.train.data_to_loaders import BinUnsupervisedDataset
from synapse.probes.sae.DirectionalTopKSAE import DirectionalTopKSAE


BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "OlMo2_1b" / "directional_topk"
D_IN = 2048
EXPANSION = 16  # d_sae = 2048 * 16 = 32768
K = 64
MAX_STEPS = 10_000
BATCH_SIZE = 4
LAYER_IDX = 8

DIR_VAL_CONFIGS = [
    (0.0, 1.0),   # pure value = standard TopK
    (1.0, 0.0),   # pure directional
    (0.5, 0.5),   # hybrid
]


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

    sae.start_recording()
    print(f"\n--- {name} frequency recording (eval pass) ---")
    eval_metrics = simple_eval(trainer, eval_ds, BATCH_SIZE, batches=10000)
    sae.stop_recording()
    trainer.remove_hook()

    sae.print_freq_dist(save_dir)
    results[name] = eval_metrics


def main():
    global lm, train_ds, eval_ds

    meta = json.loads((BIN_DIR / "meta.json").read_text())
    train_ds = BinUnsupervisedDataset(str(BIN_DIR / "train.bin"), torch.int32, tuple(meta["train_shape"]))
    eval_ds = BinUnsupervisedDataset(str(BIN_DIR / "eval.bin"), torch.int32, tuple(meta["eval_shape"]))
    lm, _, _ = build_model_and_tokenizer()

    results = {}

    for dir_c, val_c in DIR_VAL_CONFIGS:
        print(f"\n{'='*60}")
        print(f"=== directionalTopK dir={dir_c} val={val_c} k={K} ===")
        print(f"{'='*60}")
        _run(f"k_{K}/dir_{dir_c}_val_{val_c}",
             DirectionalTopKSAE(D_IN, EXPANSION, k=K, direction_coeff=dir_c, val_coeff=val_c),
             results)

    print(f"\n{'='*60}")
    print("=== COMPARISON ===")
    print(f"{'='*60}")
    header = f"{'dir':>5} {'val':>5} {'recon_pct':>10} {'active':>10}"
    print(header)
    print("-" * len(header))
    for dir_c, val_c in DIR_VAL_CONFIGS:
        name = f"k_{K}/dir_{dir_c}_val_{val_c}"
        m = results.get(name, {})
        recon = m.get("recon_pct", 0)
        active = m.get("active", 0)
        print(f"{dir_c:>5.1f} {val_c:>5.1f} {recon:>10.1f} {active:>10.1f}")


if __name__ == "__main__":
    main()
