# weakly causal crosscoder (Anthropic, "Sparse Crosscoders" 2024, §3.3.2) trained with Anthropic's JumpReLU recipe
# (Circuits Updates Jan 2025, with the CLT changes from "Circuit Tracing" 2025).
# every feature belongs to one encoder layer i: it reads the residual of layer i alone and writes to the residuals of layer
# i and every later layer. all features add up: x_hat_j = b_dec[j] + sum over layers i <= j of a_i @ W_dec[i][:, j - i].
#
# activation: a = h * 1[h > theta], h = x_i @ W_enc + b_enc, theta = exp(log_threshold) per feature. straight-through
#             gradient: da/dh = 1[h > theta], da/dlog_threshold = -theta / eps inside |h - theta| < eps / 2.
# loss:       sum_j ||x_hat_j - x_j||^2 + lam * sum_f tanh(c * ||W_dec_f|| * a_f) + lam_p * sum_f relu(theta_f - h_f) * ||W_dec_f||
#             (per token, averaged over tokens). ||W_dec_f|| is the norm of the feature's whole row over all target layers;
#             the decoder is unconstrained. lam ramps linearly from 0 over all of training (set by the trainer).
# norm:       every layer's activations are scaled by its own norm_factor (mean L2 norm sqrt(d_in)), inputs and targets alike.
# shards:     a WCCShard holds the features of a subset of encoder layers so they can be split across GPUs (b_dec lives in
#             one shard). its decode returns the partial reconstruction of its own features; the full x_hat is the sum over
#             shards. a single shard owning every layer and b_dec is the whole crosscoder.
# decode:     dense bf16 matmul while many features fire; once the batch's mean active count per token drops below
#             SPARSE_BELOW the forward and the decoder gradient only touch active rows (embedding_bag). the gradient into the
#             activations stays dense: the straight-through estimator needs it for features just below threshold.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 21
THETA_INIT = 0.03  # CLT paper: initial JumpReLU threshold
BANDWIDTH = 1.0  # CLT paper: straight-through bandwidth eps
C = 4.0  # tanh sharpness
LAM_P = 3e-6  # pre-activation loss coefficient
# mean active features per token (one encoder layer) under which decoding switches to the active rows only
SPARSE_BELOW = 16
# share of all features (every layer) firing per token right after init: b_enc is set so each fires 10000 / total features
INIT_FIRING = 10000


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, log_threshold):
        # h (N, F) pre-activations, log_threshold (F,) -> (N, F)
        theta = log_threshold.exp()
        ctx.save_for_backward(h, theta)
        return h * (h > theta)

    @staticmethod
    def backward(ctx, g):
        h, theta = ctx.saved_tensors
        # (N, F) rectangle kernel around the threshold
        near = ((h - theta).abs() < BANDWIDTH / 2).to(g.dtype)
        return g * (h > theta), (g * near).sum(0) * (-theta / BANDWIDTH)


class Decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, W):
        # a (N, F) activations (mostly zero), W (F, M) decoder -> (N, M) in fp32
        sparse = (a > 0).sum().item() / a.shape[0] < SPARSE_BELOW
        ctx.sparse = sparse
        if sparse:
            rows, cols = a.nonzero(as_tuple=True)
            # (N + 1,) bag boundaries: token n's active features are cols[offsets[n]:offsets[n + 1]]
            offsets = torch.zeros(a.shape[0] + 1, dtype=torch.long, device=a.device)
            offsets[1:] = torch.bincount(rows, minlength=a.shape[0]).cumsum(0)
            vals = a[rows, cols]
            ctx.save_for_backward(a, W, cols, offsets, vals)
            return F.embedding_bag(cols, W, offsets[:-1], per_sample_weights=vals, mode="sum")
        ctx.save_for_backward(a, W)
        return (a.bfloat16() @ W.bfloat16()).float()

    @staticmethod
    def backward(ctx, g):
        # (N, M) -> (N, F) dense: the straight-through estimator needs it for inactive features near their threshold
        da = (g.bfloat16() @ ctx.saved_tensors[1].bfloat16().T).float()
        if ctx.sparse:
            _, W, cols, offsets, vals = ctx.saved_tensors
            with torch.enable_grad():
                W_ = W.detach().requires_grad_()
                out = F.embedding_bag(cols, W_, offsets[:-1], per_sample_weights=vals, mode="sum")
                dW, = torch.autograd.grad(out, W_, g)
            return da, dW
        a = ctx.saved_tensors[0]
        # (F, N) @ (N, M) -> (F, M)
        return da, (a.bfloat16().T @ g.bfloat16()).float()


class WCCShard(nn.Module):
    def __init__(self, layers, n_layers, d_in, n_features, owns_bias=False):
        # layers: encoder layers this shard owns, n_features: features per encoder layer
        super().__init__()
        self.layers = sorted(layers)
        self.n_layers = n_layers
        self.d_in = d_in
        self.n_features = n_features
        self.owns_bias = owns_bias
        self.W_enc = nn.ParameterDict()  # {str(i): (d_in, n_features)}
        self.b_enc = nn.ParameterDict()  # {str(i): (n_features,)}
        self.log_threshold = nn.ParameterDict()  # {str(i): (n_features,)}
        self.W_dec = nn.ParameterDict()  # {str(i): (n_features, (n_layers - i) * d_in)} flattened over target layers i..
        enc_bound = 1 / math.sqrt(n_features)
        dec_bound = 1 / math.sqrt(n_layers * d_in)
        for i in self.layers:
            # init depends only on the layer, so any split of layers over shards gives the same crosscoder
            g = torch.Generator().manual_seed(SEED + i)
            self.W_enc[str(i)] = nn.Parameter((torch.rand(d_in, n_features, generator=g) * 2 - 1) * enc_bound)
            self.b_enc[str(i)] = nn.Parameter(torch.zeros(n_features))
            self.log_threshold[str(i)] = nn.Parameter(torch.full((n_features,), math.log(THETA_INIT)))
            self.W_dec[str(i)] = nn.Parameter((torch.rand(n_features, (n_layers - i) * d_in, generator=g) * 2 - 1) * dec_bound)
            # tokens since each feature last fired (stats only)
            self.register_buffer(f"tokens_since_fired_{i}", torch.zeros(n_features, dtype=torch.long))
        if owns_bias:
            # (n_layers, d_in)
            self.b_dec = nn.Parameter(torch.zeros(n_layers, d_in))
        # (n_layers,) per-layer input scaling, set by the trainer
        self.register_buffer("norm_factor", torch.ones(n_layers))

    def pre(self, x, i):
        # x (N, d_in) scaled residual of layer i -> (N, n_features) pre-activations
        return x @ self.W_enc[str(i)] + self.b_enc[str(i)]

    @torch.no_grad()
    def init_b_enc(self, x, total_features):
        # x (N, n_layers, d_in) scaled residuals: per feature, b_enc so it fires on INIT_FIRING / total_features of tokens
        N = x.shape[0]
        k = N - max(1, round(N * INIT_FIRING / total_features))
        for i in self.layers:
            # (n_features,) the (1 - p) quantile of each feature's pre-activation
            q = torch.kthvalue(self.pre(x[:, i], i), k, dim=0).values
            self.b_enc[str(i)] += THETA_INIT - q

    def decode_partial(self, x):
        # x (N, n_layers, d_in) scaled residuals -> x_hat (N, n_layers, d_in) this shard's share of the reconstruction,
        # codes {i: (h (N, F) pre-activations, a (N, F) activations)}
        N, L, D = x.shape
        x_hat = self.b_dec.expand(N, L, D).clone() if self.owns_bias else x.new_zeros(N, L, D)
        codes = {}  # {layer: (h, a)}
        for i in self.layers:
            h = self.pre(x[:, i], i)
            a = JumpReLU.apply(h, self.log_threshold[str(i)])
            codes[i] = (h, a)
            # (N, F) -> (N, (L - i) * D) -> (N, L - i, D)
            x_hat[:, i:] += Decode.apply(a, self.W_dec[str(i)]).view(N, L - i, D)
        return x_hat, codes

    def sparsity_losses(self, codes, lam):
        # summed tanh sparsity loss (times lam) and pre-activation loss over this shard's features, per token
        sp = 0.0
        pa = 0.0
        for i, (h, a) in codes.items():
            # (F,) whole-row decoder norms
            dn = self.W_dec[str(i)].norm(dim=1)
            sp = sp + torch.tanh(C * dn * a).sum(-1).mean()
            pa = pa + (torch.relu(self.log_threshold[str(i)].exp() - h) * dn).sum(-1).mean()
        return lam * sp, LAM_P * pa

    @torch.no_grad()
    def track_fired(self, codes):
        # updates the tokens-since-fired counters -> {layer: active features per token (mean)}, {layer: dead fraction
        # (no fire in dead_tokens)}
        l0 = {}  # {layer: float}
        dead = {}  # {layer: float}
        for i, (_, a) in codes.items():
            counter = getattr(self, f"tokens_since_fired_{i}")
            active = a > 0
            counter += a.shape[0]
            counter[active.any(0)] = 0
            l0[i] = active.sum(-1).float().mean().item()
            dead[i] = (counter > 10_000_000).float().mean().item()
        return l0, dead
