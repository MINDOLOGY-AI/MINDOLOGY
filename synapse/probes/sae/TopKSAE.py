# TopK SAE (Gao et al. 2024): each token keeps its k largest pre-activations, no sparsity penalty.
# follows dictionary_learning's TopK trainer so results are comparable with SAEBench.
#
# aux loss (AuxK): dead features (no fire for dead_tokens tokens) get to reconstruct the residual through their
#                  own top aux_k, scaled by aux_coeff — keeps the dictionary alive.
# activations are scaled by norm_factor (set by the trainer so the mean L2 norm is sqrt(d_in)); encode/decode
# take and return activations in the model's own scale.
# limited decoder: the top-k is kept as (vals, idx) and decoding sums only those k decoder rows (embedding_bag),
# never a dense (N, d_sae) feature vector. training decodes with a triton embedding_bag (embedding_bag_triton.py, ~2.7x
# faster backward); decode() keeps torch's, which also runs on cpu.
# compute_loss never syncs with the cpu (no .item(), no boolean-mask indexing, no data-dependent shapes), so the
# training loop can queue kernels ahead of the gpu; metrics come back as 0-dim gpu tensors.

import torch
import torch.nn as nn
import torch.nn.functional as F

from synapse.probes.sae.embedding_bag_triton import embedding_bag


class TopKSAE(nn.Module):
    def __init__(self, embed_dim, expansion_factor, k=64, aux_k=512, aux_coeff=1 / 32, dead_tokens=10_000_000):
        super().__init__()
        self.d_in = embed_dim
        self.d_sae = embed_dim * expansion_factor
        self.k = k
        self.aux_k = aux_k
        self.aux_coeff = aux_coeff
        self.dead_tokens = dead_tokens

        # encoder = decoder^T at init (dictionary_learning default), unit-norm decoder rows
        # (d_sae, d_in)
        W_dec = torch.randn(self.d_sae, self.d_in)
        W_dec /= W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        # (d_in, d_sae)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(self.d_in))
        # scaling of inputs; per-feature tokens since last fire
        self.register_buffer("norm_factor", torch.tensor(1.0))
        self.register_buffer("tokens_since_fired", torch.zeros(self.d_sae, dtype=torch.long))

    def _norm_decoder(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def encode(self, x):
        # x: (N, d_in) model-scale activations -> vals (N, k) relu'd top-k pre-activations (strongest first),
        # idx (N, k) their feature ids
        # (N, d_in) -> (N, d_sae)
        pre = torch.relu((x * self.norm_factor - self.b_dec) @ self.W_enc + self.b_enc)
        vals, idx = torch.topk(pre, self.k, dim=-1)
        return vals, idx

    def decode(self, vals, idx):
        # vals, idx: (N, k) -> (N, d_in) model-scale reconstruction from only the k chosen decoder rows
        return (F.embedding_bag(idx, self.W_dec, per_sample_weights=vals, mode="sum") + self.b_dec) / self.norm_factor

    def compute_loss(self, batch_data):
        # batch_data: (N, d_in) model-scale activations
        self._norm_decoder()
        x = batch_data * self.norm_factor
        # (N, d_in) -> (N, d_sae)
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        # (N, k), (N, k)
        vals, idx = torch.topk(pre, self.k, dim=-1)
        # (N, k) -> (N, d_in), scaled space
        x_hat = embedding_bag(idx, self.W_dec, vals) + self.b_dec
        recon_loss = (x_hat - x).pow(2).sum(-1).mean()

        # dead-feature bookkeeping
        with torch.no_grad():
            # (d_sae,) features with a positive value on any token of the batch
            fired = torch.zeros(self.d_sae, device=x.device).scatter_reduce_(0, idx.flatten(), (vals > 0).flatten().float(), "amax") > 0
            self.tokens_since_fired += x.shape[0]
            self.tokens_since_fired.masked_fill_(fired, 0)
        # (d_sae,)
        dead = self.tokens_since_fired > self.dead_tokens

        # AuxK: dead features reconstruct the residual through their own top aux_k. always the full aux_k with live
        # features zeroed: past the dead ones the top-k picks zeros, which add nothing and get no gradient, so this
        # equals a top-min(aux_k, n_dead) without reading n_dead on the cpu; with nothing dead the loss is masked to 0
        # (N, d_in)
        residual = (x - x_hat).detach()
        # (N, d_sae) with live features zeroed
        pre_dead = torch.where(dead.unsqueeze(0), pre, torch.zeros_like(pre))
        # (N, aux_k), (N, aux_k)
        aux_vals, aux_idx = torch.topk(pre_dead, self.aux_k, dim=-1)
        # (N, aux_k) -> (N, d_in)
        aux_recon = embedding_bag(aux_idx, self.W_dec, aux_vals)
        aux_loss = self.aux_coeff * (aux_recon - residual).pow(2).sum(-1).mean() * dead.any()

        metrics = {
            "recon_err_pct": (recon_loss / x.pow(2).sum(-1).mean() * 100).detach(),
            "l0": (vals > 0).float().sum(-1).mean(),
            "dead_frac": dead.float().mean(),
            "aux": aux_loss.detach(),
        }
        return recon_loss + aux_loss, metrics
