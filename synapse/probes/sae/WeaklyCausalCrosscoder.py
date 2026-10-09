# weakly causal crosscoder (Anthropic, "Sparse Crosscoders" 2024, §3.3.2) with TopK activations.
# every feature belongs to one encoder layer i: it reads the residual of layer i alone and its decoder reconstructs the
# residuals of layer i and every later layer. x_hat_j = sum over features of layers i <= j of a * W_dec[i][f, j - i] + b_dec[j].
# a per-layer TopK SAE is the special case where the decoder stops at layer i.
#
# topk:   each encoder layer keeps its own k largest pre-activations per token (k per layer, like the per-layer SAEs).
# norm:   every layer's activations are scaled by its own norm_factor (mean L2 norm sqrt(d_in)), inputs and targets alike;
#         each feature's decoder is unit norm over its whole (n_layers - i) * d_in row.
# AuxK:   dead features of layer i (no fire for dead_tokens tokens) reconstruct the layer-i residual error through the
#         layer-i slice of their decoder, scaled by aux_coeff.
# shards: a WCCShard holds the features of a subset of encoder layers so they can be split across GPUs (b_dec lives in one
#         shard). its decode returns the partial reconstruction of its own features; the full x_hat is the sum over shards.
#         a single shard owning every layer and b_dec is the whole crosscoder.
# limited decoder: decoding sums only the k chosen decoder rows per layer (embedding_bag), never a dense feature vector.

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 21


class WCCShard(nn.Module):
    def __init__(self, layers, n_layers, d_in, n_features, k=64, aux_k=512, aux_coeff=1 / 32, dead_tokens=10_000_000,
                 owns_bias=False):
        # layers: encoder layers this shard owns, n_features: features per encoder layer
        super().__init__()
        self.layers = sorted(layers)
        self.n_layers = n_layers
        self.d_in = d_in
        self.n_features = n_features
        self.k = k
        self.aux_k = aux_k
        self.aux_coeff = aux_coeff
        self.dead_tokens = dead_tokens
        self.owns_bias = owns_bias
        self.W_enc = nn.ParameterDict()  # {str(i): (d_in, n_features)}
        self.b_enc = nn.ParameterDict()  # {str(i): (n_features,)}
        self.W_dec = nn.ParameterDict()  # {str(i): (n_features, (n_layers - i) * d_in)} flattened over target layers i..
        for i in self.layers:
            # init depends only on the layer, so any split of layers over shards gives the same crosscoder
            g = torch.Generator().manual_seed(SEED + i)
            # (n_features, n_layers - i, d_in), unit norm over each feature's whole row
            W = torch.randn(n_features, n_layers - i, d_in, generator=g)
            W /= W.flatten(1).norm(dim=1)[:, None, None]
            # encoder = own-layer decoder slice^T, unit-norm columns
            # (n_features, d_in)
            own = W[:, 0, :]
            self.W_enc[str(i)] = nn.Parameter((own / own.norm(dim=1, keepdim=True)).T.contiguous())
            self.b_enc[str(i)] = nn.Parameter(torch.zeros(n_features))
            self.W_dec[str(i)] = nn.Parameter(W.flatten(1))
            self.register_buffer(f"tokens_since_fired_{i}", torch.zeros(n_features, dtype=torch.long))
        if owns_bias:
            # (n_layers, d_in)
            self.b_dec = nn.Parameter(torch.zeros(n_layers, d_in))
        # (n_layers,) per-layer input scaling, set by the trainer
        self.register_buffer("norm_factor", torch.ones(n_layers))

    def norm_decoder(self):
        with torch.no_grad():
            for W in self.W_dec.values():
                W /= W.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def encode(self, x, i):
        # x: (N, d_in) scaled residual of layer i -> pre (N, n_features), vals (N, k) relu'd top-k, idx (N, k) feature ids
        # (N, d_in) -> (N, n_features)
        pre = torch.relu(x @ self.W_enc[str(i)] + self.b_enc[str(i)])
        vals, idx = torch.topk(pre, self.k, dim=-1)
        return pre, vals, idx

    def decode_partial(self, x):
        # x: (N, n_layers, d_in) scaled residuals -> x_hat (N, n_layers, d_in) this shard's share of the reconstruction,
        # codes {i: (pre, vals, idx)}
        N, L, D = x.shape
        x_hat = self.b_dec.expand(N, L, D).clone() if self.owns_bias else x.new_zeros(N, L, D)
        codes = {}  # {layer: (pre (N, n_features), vals (N, k), idx (N, k))}
        for i in self.layers:
            pre, vals, idx = self.encode(x[:, i], i)
            codes[i] = (pre, vals, idx)
            # (N, k) -> (N, (L - i) * D) -> (N, L - i, D)
            x_hat[:, i:] += F.embedding_bag(idx, self.W_dec[str(i)], per_sample_weights=vals, mode="sum").view(N, L - i, D)
        return x_hat, codes

    def aux_loss(self, x, x_hat_full, codes):
        # x, x_hat_full: (N, n_layers, d_in) scaled residuals and the full (all-shard) reconstruction, codes from
        # decode_partial. updates the dead-feature counters; returns the summed AuxK loss and {layer: dead fraction}
        N = x.shape[0]
        total = x.new_zeros(())
        dead_frac = {}  # {layer: float}
        for i, (pre, vals, idx) in codes.items():
            counter = getattr(self, f"tokens_since_fired_{i}")
            # (n_features,) features with a positive value on any token of the batch
            fired = torch.zeros(self.n_features, dtype=torch.bool, device=x.device)
            fired[idx[vals > 0]] = True
            with torch.no_grad():
                counter += N
                counter[fired] = 0
            # (n_features,)
            dead = counter > self.dead_tokens
            n_dead = int(dead.sum())
            dead_frac[i] = n_dead / self.n_features
            if n_dead == 0:
                continue
            # (N, d_in)
            residual = (x[:, i] - x_hat_full[:, i]).detach()
            # (N, n_features) with live features zeroed
            pre_dead = torch.where(dead.unsqueeze(0), pre, torch.zeros_like(pre))
            # (N, aux), (N, aux)
            aux_vals, aux_idx = torch.topk(pre_dead, min(self.aux_k, n_dead), dim=-1)
            # (n_features, d_in) the layer-i slice of every decoder row
            W_own = self.W_dec[str(i)][:, :self.d_in]
            # (N, aux) -> (N, d_in)
            aux_recon = F.embedding_bag(aux_idx, W_own, per_sample_weights=aux_vals, mode="sum")
            total = total + self.aux_coeff * (aux_recon - residual).pow(2).sum(-1).mean()
        return total, dead_frac
