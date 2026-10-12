# inference side of a trained weakly causal crosscoder (WeaklyCausalCrosscoder.py, trained by
# evoke/Qwen3_5_4b/interp/wcc_resid_jumprelu.py): one encoder layer's features, without the training machinery
# (no bitsandbytes), so the labeler and interpviz can load it anywhere.
# it exposes the attributes a TopK SAE has (W_dec, norm_factor, b_dec, encode), so the SAE attribution and edit code
# (synapse/interp/sae_attribute.py, sae_intervene.py) runs on it unchanged: W_dec is each feature's write into its own
# layer (the one it reads), and an edit there reaches the later layers through the model itself.

import torch
import torch.nn as nn


class WCCEncoder(nn.Module):
    # one encoder layer of a trained crosscoder, for inference (picks, interpviz). loads the trainer's final
    # L<layer>.pt (W_enc, b_enc, log_threshold, W_dec (F, n_layers - layer, d_in)) and shared.pt (b_dec, norm_factor).
    # the decoder stays memory-mapped on disk (GBs per layer): only the rows asked for are read.
    # k: how many active features encode() returns per token (strongest first)
    def __init__(self, weights_path, shared_path, layer, k=64):
        super().__init__()
        w = torch.load(weights_path, map_location="cpu", mmap=True, weights_only=True)
        shared = torch.load(shared_path, map_location="cpu", weights_only=True)
        self.k = k
        # (F, n_layers - layer, d_in) scaled-space writes into this layer and every later one, memory-mapped
        self.W_dec_all = w["W_dec"]
        # (F, d_in) scaled-space write into this layer, a view of W_dec_all
        self.W_dec = self.W_dec_all[:, 0]
        self.d_sae = w["W_enc"].shape[1]
        # (d_in, F), (F,), (F,)
        self.register_buffer("W_enc", w["W_enc"].float())
        self.register_buffer("b_enc", w["b_enc"].float())
        self.register_buffer("threshold", w["log_threshold"].float().exp())
        # () this layer's scale (model -> scaled space), (d_in,) the crosscoder's constant in this layer, scaled space
        self.register_buffer("norm_factor", shared["norm_factor"][layer].float().clone())
        self.register_buffer("b_dec", shared["b_dec"][layer].float().clone())
        # (n_layers - layer,) the scale of every layer it writes to (the decoder writes in scaled space)
        self.out_scale = shared["norm_factor"][layer:].float().clone()

    def acts(self, x):
        # x (N, d_in) model-scale residual of this layer -> (N, F) JumpReLU activations
        h = (x.float() * self.norm_factor) @ self.W_enc + self.b_enc
        return h * (h > self.threshold)

    def encode(self, x):
        # x (N, d_in) -> vals (N, k) strongest activations first (zeros where fewer than k fire), idx (N, k) feature ids
        return torch.topk(self.acts(x), self.k, dim=-1)

    def out_dir(self, unit, target):
        # feature `unit`'s model-scale write into layer (this layer + target), target -1 = the last layer -> (d_in,) fp32
        return self.W_dec_all[unit, target].float() / self.out_scale[target]
