# HookedModel that exposes SAE feature activations on OlMo2's resid_post.<layer> as hook names,
# so the picks tracker / labeler can treat SAE features exactly like MLP neurons.

import torch

from synapse.interp.hooked_model import HookedModel


class HookedOlmoSAE(HookedModel):
    def __init__(self, model, layer, saes):
        # saes: {name: sae module in eval mode, on the model's device}
        self.layer = layer
        self.saes = saes
        super().__init__(model)

    def hook_points(self):
        return {"resid": (self.model.model.layers[self.layer], False)}

    def run(self, tokens, names):
        # names: sae names -> self.acts[name] = (b, L, d_sae) features
        super().run(tokens, ["resid"])
        resid = self.acts["resid"]  # (b, L, d_in)
        b, L, d = resid.shape
        with torch.no_grad():
            for name in names:
                self.acts[name] = self.saes[name].encode(resid.reshape(-1, d).float()).view(b, L, -1)
