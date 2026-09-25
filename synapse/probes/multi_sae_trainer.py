# trains one SAE per hooked layer in a single LM pass.
# the LM is a frozen backbone inside compute_loss — simple_train sees
# one module, one summed loss, one flat metrics dict (keys prefixed per layer).

import torch
import torch.nn as nn


class _StopForward(Exception):
    # raised by the deepest capture hook: layers past the last hooked one are never run
    pass


class MultiSAETrainer(nn.Module):
    def __init__(self, inner_lm, saes):
        # inner_lm: callable with input_ids=, must expose .layers (block modules)
        # saes: {layer_idx: sae_module}
        super().__init__()
        # list wrapper keeps the frozen LM out of the module registry,
        # so checkpoints and param groups only contain SAE weights
        self._lm = [inner_lm]
        inner_lm.cuda()
        self.saes = nn.ModuleDict({str(i): sae for i, sae in saes.items()})
        self.layer_indices = list(saes.keys())
        self.lm.eval()
        for p in self.lm.parameters():
            p.requires_grad = False
        self._acts = {}
        self._hooks = [
            self.lm.layers[i].register_forward_hook(self._make_capture(i))
            for i in self.layer_indices
        ]

    @property
    def lm(self):
        return self._lm[0]

    def _make_capture(self, layer_idx):
        def capture(module, input, output):
            self._acts[layer_idx] = output[0].detach()  # (B, seq, d_model)
            if layer_idx == max(self.layer_indices):
                raise _StopForward
        return capture

    def run(self, token_chunk):
        # LM forward up to the deepest hooked layer; fills self._acts {layer_idx: (B, seq, d_model)}
        with torch.no_grad():
            try:
                self.lm(input_ids=token_chunk)
            except _StopForward:
                pass

    def compute_loss(self, token_chunk):
        self.run(token_chunk)
        total = 0.0
        metrics = {}
        for i in self.layer_indices:
            acts = self._acts[i].reshape(-1, self._acts[i].shape[-1]).float()  # (B*seq, d_model)
            loss, m = self.saes[str(i)].compute_loss(acts)
            total = total + loss
            for k, v in m.items():
                metrics[f"l{i}_{k}"] = v
        return total, metrics

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def train(self, mode=True):
        super().train(mode)
        self.lm.eval()
        return self
