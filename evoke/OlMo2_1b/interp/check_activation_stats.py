import json
import torch
import torch.nn as nn
from pathlib import Path

from evoke.OlMo2_1b.run.loader import build_model_and_tokenizer
from synapse.train.data_to_loaders import BinUnsupervisedDataset
from synapse.train.simple_train import _to_cuda

BIN_DIR = Path.cwd() / "data" / "datasteps" / "tokenized" / "olmo2_1b_interp_dataset"


def main():
    meta = json.loads((BIN_DIR / "meta.json").read_text())
    ds = BinUnsupervisedDataset(str(BIN_DIR / "train.bin"), torch.int32, tuple(meta["train_shape"]))
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

    lm, _, _ = build_model_and_tokenizer()

    _acts = None

    def hook(m, i, o):
        nonlocal _acts
        _acts = o[0].detach()

    lm.model.layers[8].register_forward_hook(hook)

    norms = []  # [float] L2 norm per token
    stds = []  # [float] scalar std across all dims+token
    means = []  # [float] scalar mean

    for i, batch in enumerate(dl):
        if i >= 50:
            break
        batch = _to_cuda(batch)
        with torch.no_grad():
            lm.model(input_ids=batch)
        acts = _acts.reshape(-1, 2048)  # (B*seq, 2048)
        norms.append(acts.norm(dim=-1).mean().item())
        stds.append(acts.std().item())
        means.append(acts.mean().item())
        if i % 10 == 0:
            print(f"  batch {i}/50")

    avg_norm = sum(norms) / len(norms)
    avg_std = sum(stds) / len(stds)
    avg_mean = sum(means) / len(means)

    print(f"\n--- layer 8 residual stream stats ---")
    print(f"L2 norm per token:  {avg_norm:.1f}")
    print(f"Std (scalar):       {avg_std:.2f}")
    print(f"Mean (scalar):      {avg_mean:.3f}")
    print(f"Std/mean ratio:     {avg_std / abs(avg_mean):.1f}" if abs(avg_mean) > 0.001 else "mean ≈ 0 — std is already clean")
    print(f"RMS per element:    {avg_norm / (2048 ** 0.5):.3f}  (norm / sqrt(dim))")


if __name__ == "__main__":
    main()
