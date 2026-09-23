# TopK-family SAE: TopK (per token), BatchTopK, and Matryoshka-BatchTopK in one module
# (Gao et al. 2024 TopK; Bussmann et al. 2024 BatchTopK; Bussmann et al. 2025 Matryoshka), following
# dictionary_learning's trainers so results are comparable with SAEBench.
#
# per_token=True : classic TopK — each token keeps its k largest pre-activations.
# per_token=False: BatchTopK — the batch of N tokens keeps the k*N largest pre-activations overall, so
#                  tokens get a variable number of features (avg k). at eval a JumpReLU threshold (ema of the
#                  smallest selected activation during training) replaces the batch selection.
# matryoshka_fractions: nested prefix dictionaries, e.g. [1/32, 1/32, 1/16, 1/8, 1/4, 1/2]; the loss is the sum
#                  of the reconstruction losses of every prefix, which stops big features absorbing small ones.
# aux loss (AuxK): dead features (no fire for dead_tokens tokens) get to reconstruct the residual through their
#                  own top aux_k, scaled by aux_coeff — keeps the dictionary alive without a sparsity penalty.
# activations are scaled by norm_factor (set by the trainer so the mean L2 norm is sqrt(d_in)); encode/decode
# take and return activations in the model's own scale.

import torch
import torch.nn as nn

THRESHOLD_EMA = 0.999


class BatchTopKSAE(nn.Module):
    def __init__(self, embed_dim, expansion_factor, k=64, per_token=False, matryoshka_fractions=None,
                 aux_k=512, aux_coeff=1 / 32, dead_tokens=10_000_000):
        super().__init__()
        self.d_in = embed_dim
        self.d_sae = embed_dim * expansion_factor
        self.k = k
        self.per_token = per_token
        self.aux_k = aux_k
        self.aux_coeff = aux_coeff
        self.dead_tokens = dead_tokens
        # [1024, 2048, 4096, ...] cumulative prefix sizes of the nested dictionaries (last = d_sae), or None
        self.prefixes = None
        if matryoshka_fractions is not None:
            assert abs(sum(matryoshka_fractions) - 1) < 1e-6, "matryoshka fractions must sum to 1"
            self.prefixes = [int(round(sum(matryoshka_fractions[:i + 1]) * self.d_sae)) for i in range(len(matryoshka_fractions))]
            self.prefixes[-1] = self.d_sae

        # encoder = decoder^T at init (dictionary_learning default), unit-norm decoder rows
        W_dec = torch.randn(self.d_sae, self.d_in)
        W_dec /= W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))
        # eval-time JumpReLU threshold for batch mode; scaling of inputs; per-feature tokens since last fire
        self.register_buffer("threshold", torch.tensor(-1.0))
        self.register_buffer("norm_factor", torch.tensor(1.0))
        self.register_buffer("tokens_since_fired", torch.zeros(self.d_sae, dtype=torch.long))

    def _norm_decoder(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def _select(self, pre):
        # pre: (N, d_sae) relu'd pre-activations -> (N, d_sae) sparse features (train-time selection rule)
        n = pre.shape[0]
        if self.per_token:
            _, idx = torch.topk(pre, self.k, dim=-1)  # (N, k)
            return torch.zeros_like(pre).scatter(-1, idx, pre.gather(-1, idx))
        flat = pre.flatten()
        _, idx = torch.topk(flat, self.k * n)  # (k*N,)
        return torch.zeros_like(flat).scatter(0, idx, flat[idx]).view_as(pre)

    def encode(self, x):
        # x: (N, d_in) model-scale activations -> (N, d_sae) features. eval uses the learned threshold in batch mode
        pre = torch.relu((x * self.norm_factor - self.b_dec) @ self.W_enc + self.b_enc)
        if self.per_token or self.training:
            return self._select(pre)
        assert self.threshold >= 0, "batch-topk threshold not learned yet"
        return pre * (pre > self.threshold)

    def decode(self, f):
        # f: (N, d_sae) -> (N, d_in) model-scale reconstruction
        return (f @ self.W_dec + self.b_dec) / self.norm_factor

    def compute_loss(self, batch_data):
        # batch_data: (N, d_in) model-scale activations
        self._norm_decoder()
        x = batch_data * self.norm_factor
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)  # (N, d_sae)
        f = self._select(pre)
        x_hat = f @ self.W_dec + self.b_dec

        if self.prefixes is None:
            recon_loss = (x_hat - x).pow(2).sum(-1).mean()
        else:
            # every nested prefix dictionary must reconstruct on its own
            recon_loss = sum((f[:, :m] @ self.W_dec[:m] + self.b_dec - x).pow(2).sum(-1).mean() for m in self.prefixes)

        # bookkeeping: firing, dead features, batch-mode threshold
        fired = (f > 0).any(dim=0)  # (d_sae,)
        with torch.no_grad():
            self.tokens_since_fired += x.shape[0]
            self.tokens_since_fired[fired] = 0
            if not self.per_token:
                min_sel = f[f > 0].min() if fired.any() else self.threshold
                self.threshold.copy_(min_sel if self.threshold < 0 else THRESHOLD_EMA * self.threshold + (1 - THRESHOLD_EMA) * min_sel)
        dead = self.tokens_since_fired > self.dead_tokens  # (d_sae,)

        # AuxK: dead features reconstruct the residual through their own top-k
        aux_loss = x.new_zeros(())
        n_dead = int(dead.sum())
        if n_dead > 0:
            residual = (x - x_hat).detach()
            pre_dead = torch.where(dead.unsqueeze(0), pre, torch.zeros_like(pre))
            _, idx = torch.topk(pre_dead, min(self.aux_k, n_dead), dim=-1)
            f_aux = torch.zeros_like(pre).scatter(-1, idx, pre_dead.gather(-1, idx))
            aux_loss = self.aux_coeff * (f_aux @ self.W_dec - residual).pow(2).sum(-1).mean()

        loss = recon_loss + aux_loss
        mse = (x_hat - x).pow(2).sum(-1).mean()
        metrics = {
            "mse": mse.item(),
            "expl_var": (1 - mse / x.var(dim=0).sum()).item(),
            "l0": (f > 0).float().sum(-1).mean().item(),
            "dead_frac": n_dead / self.d_sae,
            "aux": aux_loss.item(),
        }
        return loss, metrics
