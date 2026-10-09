# HookedModel that exposes SAE feature activations on OlMo2 residual streams as hook names,
# so the picks tracker / labeler can treat SAE features exactly like MLP neurons.

import torch

from synapse.interp.hooked_model import HookedModel


class HookedOlmoSAE(HookedModel):
    def __init__(self, model, saes):
        # saes: {name: (layer, sae module in eval mode on the model's device)}; the sae reads resid_post.<layer>
        self.saes = saes
        super().__init__(model)

    def hook_points(self):
        return {f"resid.{layer}": (self.model.model.layers[layer], False) for layer, _ in self.saes.values()}

    def run(self, tokens, names):
        # names: sae names -> self.acts[name] = (b, L, d_sae) features
        super().run(tokens, sorted({f"resid.{self.saes[n][0]}" for n in names}))
        with torch.no_grad():
            for name in names:
                layer, sae = self.saes[name]
                resid = self.acts[f"resid.{layer}"]  # (b, L, d_in)
                b, L, d = resid.shape
                # (b*L, d_in) -> (b*L, k), (b*L, k)
                vals, idx = sae.encode(resid.reshape(-1, d).float())
                # (b*L, k) -> (b, L, d_sae): the picks tracker needs every feature's value
                self.acts[name] = torch.zeros(b * L, sae.d_sae, device=vals.device).scatter_(-1, idx, vals).view(b, L, -1)
