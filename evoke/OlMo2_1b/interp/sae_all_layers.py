# trains a plain SAE on every layer of OlMo2-1B in one LM pass

import json
import torch
from pathlib import Path

from evoke.OlMo2_1b.run.loader import build_model_and_tokenizer
from synapse.train.simple_train import simple_train, simple_eval
from synapse.train.data_to_loaders import BinUnsupervisedDataset
from synapse.probes.sae.SAE import SAE
from synapse.probes.multi_sae_trainer import MultiSAETrainer

BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"
WEIGHTS_DIR = Path.cwd() / "weights" / "evoke" / "OlMo2_1b" / "sae_all_layers"
D_IN = 2048
EXPANSION = 16  # d_sae = 2048 * 16 = 32768 per layer
SPARSITY_COEFF = 1e-3
LAYERS = list(range(16))
MAX_STEPS = 10_000
BATCH_SIZE = 4
LR = 7e-5


def main():
    meta = json.loads((BIN_DIR / "meta.json").read_text())
    train_ds = BinUnsupervisedDataset(str(BIN_DIR / "train.bin"), torch.int32, tuple(meta["train_shape"]))
    eval_ds = BinUnsupervisedDataset(str(BIN_DIR / "eval.bin"), torch.int32, tuple(meta["eval_shape"]))
    lm, _, _ = build_model_and_tokenizer()

    trainer = MultiSAETrainer(lm.model, {i: SAE(D_IN, EXPANSION, SPARSITY_COEFF) for i in LAYERS})
    opt = torch.optim.Adam(trainer.parameters(), lr=LR, betas=(0.0, 0.999))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    simple_train(
        model=trainer,
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        optimizer=opt,
        epochs=1,
        save_path=str(WEIGHTS_DIR),
        batches_per_log=100,
        batches_per_save=100,
        eval_dataset=eval_ds,
        eval_batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
    )

    print("\n--- final eval ---")
    eval_metrics = simple_eval(trainer, eval_ds, BATCH_SIZE, batches=100)
    trainer.remove_hooks()
    for k, v in sorted(eval_metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
