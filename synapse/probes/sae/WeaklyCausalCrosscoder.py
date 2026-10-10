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
# decoder memory: the decoders are ~95% of the parameters, so they are not nn.Parameters. the gpu holds a bf16 working copy
#             (what every matmul reads) and a bf16 gradient accumulator; the fp32 master weights and their 8-bit adam state
#             (bitsandbytes blockwise, the same kernel as bnb.optim.Adam8bit) live in pinned cpu memory and are streamed
#             through the gpu in row chunks once per step for the update, which runs on the gpu (copies in and out overlap
#             the update of the chunk in between). the small parameters
#             (encoders, thresholds, b_dec) are ordinary nn.Parameters with an ordinary optimizer.
# decode:     dense bf16 matmul while many features fire; once the batch's mean active count per token drops below
#             SPARSE_BELOW the forward only touches active rows (embedding_bag). the decoder gradient a^T g and the gradient
#             into the activations g W^T stay dense matmuls: the straight-through estimator needs the latter for features
#             just below threshold.

import math

import bitsandbytes.functional as BF
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 21
THETA_INIT = 0.03  # CLT paper: initial JumpReLU threshold
BANDWIDTH = 1.0  # CLT paper: straight-through bandwidth eps
C = 4.0  # tanh sharpness
LAM_P = 3e-6  # pre-activation loss coefficient
# mean active features per token (one encoder layer) under which the decode forward switches to the active rows only
SPARSE_BELOW = 16
# share of all features (every layer) firing per token right after init: b_enc is set so each fires 10000 / total features
INIT_FIRING = 10000
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
QBLOCK = 256  # bitsandbytes 8-bit blockwise state: one absmax per 256 elements
CHUNK_ELEMS = 2 ** 26  # decoder elements per streamed update chunk (256MB of fp32 master)
# bitsandbytes' signed / unsigned dynamic quantization maps for adam's first / second moment (what Adam8bit uses)
QMAP1 = BF.create_dynamic_map(signed=True)
QMAP2 = BF.create_dynamic_map(signed=False)


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
    def forward(ctx, a, W, G):
        # a (N, F) activations (mostly zero), W (F, M) bf16 decoder, G (F, M) bf16 gradient accumulator -> (N, M) fp32.
        # backward adds a^T g into G (no autograd gradient for W)
        ctx.save_for_backward(a, W)
        ctx.G = G
        if (a > 0).sum().item() / a.shape[0] < SPARSE_BELOW:
            rows, cols = a.nonzero(as_tuple=True)
            # (N + 1,) bag boundaries: token n's active features are cols[offsets[n]:offsets[n + 1]]
            offsets = torch.zeros(a.shape[0] + 1, dtype=torch.long, device=a.device)
            offsets[1:] = torch.bincount(rows, minlength=a.shape[0]).cumsum(0)
            return F.embedding_bag(cols, W, offsets[:-1], per_sample_weights=a[rows, cols].bfloat16(), mode="sum").float()
        # bf16 inputs, fp32 output written directly
        return torch.mm(a.bfloat16(), W, out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, g):
        a, W = ctx.saved_tensors
        gb = g.bfloat16()
        # (F, N) @ (N, M) accumulated into the bf16 gradient buffer
        ctx.G.addmm_(a.bfloat16().T, gb)
        # (N, M) @ (M, F) -> (N, F): dense, the straight-through estimator needs it for inactive features near threshold
        return torch.mm(gb, W.T, out_dtype=torch.float32), None, None


class WCCShard(nn.Module):
    def __init__(self, layers, n_layers, d_in, n_features, device, owns_bias=False):
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
        # decoder of layer i: (n_features, (n_layers - i) * d_in), one row per feature over target layers i.. laid end to end
        self.dec_W = {}  # {i: bf16 gpu working copy}
        self.dec_G = {}  # {i: bf16 gpu gradient accumulator}
        self.dec_master = {}  # {i: fp32 pinned cpu master weights}
        self.dec_adam = {}  # {i: {"s1", "s2": (n,) uint8, "absmax1", "absmax2": (n / QBLOCK,) fp32}} pinned cpu adam state
        self.dec_norm = {}  # {i: (n_features,) fp32 gpu row norms of the master}
        self.dec_norm_leaf = {}  # {i: (n_features,) leaf copy of dec_norm the losses differentiate, reset every step}
        self.dec_step = 0  # adam steps taken by the decoders
        enc_bound = 1 / math.sqrt(n_features)
        dec_bound = 1 / math.sqrt(n_layers * d_in)
        for i in self.layers:
            # init depends only on the layer, so any split of layers over shards gives the same crosscoder
            g = torch.Generator().manual_seed(SEED + i)
            self.W_enc[str(i)] = nn.Parameter((torch.rand(d_in, n_features, generator=g) * 2 - 1).mul_(enc_bound).to(device))
            self.b_enc[str(i)] = nn.Parameter(torch.zeros(n_features, device=device))
            self.log_threshold[str(i)] = nn.Parameter(torch.full((n_features,), math.log(THETA_INIT), device=device))
            M = (n_layers - i) * d_in
            assert M % QBLOCK == 0, f"decoder row length {M} must be a multiple of the adam block {QBLOCK}"
            self.dec_master[i] = torch.empty(n_features, M, pin_memory=True).uniform_(-dec_bound, dec_bound, generator=g)
            n = n_features * M
            self.dec_adam[i] = {"s1": torch.zeros(n, dtype=torch.uint8, pin_memory=True),
                                "s2": torch.zeros(n, dtype=torch.uint8, pin_memory=True),
                                "absmax1": torch.zeros(n // QBLOCK, pin_memory=True),
                                "absmax2": torch.zeros(n // QBLOCK, pin_memory=True)}
            self.dec_W[i] = torch.empty(n_features, M, dtype=torch.bfloat16, device=device)
            self.dec_G[i] = torch.zeros(n_features, M, dtype=torch.bfloat16, device=device)
            self.dec_norm[i] = torch.empty(n_features, device=device)
            # tokens since each feature last fired (stats only)
            self.register_buffer(f"tokens_since_fired_{i}", torch.zeros(n_features, dtype=torch.long, device=device))
        self.refresh_decoders()
        # side streams and two gpu buffer sets for the pipelined decoder update (dec_update)
        self.h2d = torch.cuda.Stream(device)
        self.d2h = torch.cuda.Stream(device)
        self.update_bufs = [{"p": torch.empty(CHUNK_ELEMS, device=device),
                             "s1": torch.empty(CHUNK_ELEMS, dtype=torch.uint8, device=device),
                             "s2": torch.empty(CHUNK_ELEMS, dtype=torch.uint8, device=device),
                             "a1": torch.empty(CHUNK_ELEMS // QBLOCK, device=device),
                             "a2": torch.empty(CHUNK_ELEMS // QBLOCK, device=device)} for _ in range(2)]
        if owns_bias:
            # (n_layers, d_in)
            self.b_dec = nn.Parameter(torch.zeros(n_layers, d_in, device=device))
        # (n_layers,) per-layer input scaling, set by the trainer
        self.register_buffer("norm_factor", torch.ones(n_layers, device=device))

    def chunks(self, i):
        # row ranges of layer i's decoder, at most CHUNK_ELEMS elements each -> [(r0, r1)]
        rows = CHUNK_ELEMS // self.dec_master[i].shape[1]
        assert rows >= 1, f"a decoder row ({self.dec_master[i].shape[1]}) is longer than CHUNK_ELEMS
        return [(r, min(r + rows, self.n_features)) for r in range(0, self.n_features, rows)]

    @torch.no_grad()
    def refresh_decoders(self):
        # bf16 working copies and row norms from the masters (after init or loading)
        for i in self.layers:
            for r0, r1 in self.chunks(i):
                p = self.dec_master[i][r0:r1].to(self.dec_W[i].device)
                self.dec_W[i][r0:r1] = p.bfloat16()
                self.dec_norm[i][r0:r1] = p.norm(dim=1)

    def pre(self, x, i):
        # x (N, d_in) scaled residual of layer i -> (N, n_features) pre-activations
        return x @ self.W_enc[str(i)] + self.b_enc[str(i)]

    def scaled(self, x, i):
        # x (N, n_layers, d_in) model-scale residuals (any dtype) -> (N, d_in) fp32 layer i, scaled by its norm_factor
        return x[:, i].float() * self.norm_factor[i]

    @torch.no_grad()
    def init_b_enc(self, x, total_features):
        # x (N, n_layers, d_in) model-scale residuals: per feature, b_enc so it fires on INIT_FIRING / total_features of tokens
        N = x.shape[0]
        k = N - max(1, round(N * INIT_FIRING / total_features))
        for i in self.layers:
            # (n_features,) the (1 - p) quantile of each feature's pre-activation
            q = torch.kthvalue(self.pre(self.scaled(x, i), i), k, dim=0).values
            self.b_enc[str(i)] += THETA_INIT - q

    def decode_partial(self, x):
        # x (N, n_layers, d_in) model-scale residuals (bf16 from the LM) -> x_hat (N, n_layers, d_in) fp32 this shard's
        # share of the reconstruction in scaled space, codes {i: (h (N, F) pre-activations, a (N, F) activations)}
        N, L, D = x.shape
        x_hat = self.b_dec.expand(N, L, D).clone() if self.owns_bias else torch.zeros(N, L, D, device=x.device)
        codes = {}  # {layer: (h, a)}
        for i in self.layers:
            h = self.pre(self.scaled(x, i), i)
            a = JumpReLU.apply(h, self.log_threshold[str(i)])
            codes[i] = (h, a)
            # (N, F) -> (N, (L - i) * D) -> (N, L - i, D)
            x_hat[:, i:] += Decode.apply(a, self.dec_W[i], self.dec_G[i]).view(N, L - i, D)
        return x_hat, codes

    def begin_step(self):
        # fresh leaves for the decoder row norms: their .grad sums the sparsity / pre-act losses' d/d||W_dec_f|| over the
        # step's micro-batches
        self.dec_norm_leaf = {i: self.dec_norm[i].clone().requires_grad_() for i in self.layers}

    def sparsity_losses(self, codes, lam):
        # summed tanh sparsity loss (times lam) and pre-activation loss over this shard's features, per token
        sp = 0.0
        pa = 0.0
        for i, (h, a) in codes.items():
            # (F,) whole-row decoder norms
            dn = self.dec_norm_leaf[i]
            sp = sp + torch.tanh(C * dn * a).sum(-1).mean()
            pa = pa + (torch.relu(self.log_threshold[str(i)].exp() - h) * dn).sum(-1).mean()
        return lam * sp, LAM_P * pa

    def norm_grad_scale(self, i):
        # (F,) d loss / d||W_dec_f|| / ||W_dec_f||: the full decoder gradient is dec_G + this[:, None] * W_dec
        g = self.dec_norm_leaf[i].grad
        assert g is not None, f"layer {i}: no gradient reached the decoder norms (sparsity_losses not in the backward?)"
        return g / self.dec_norm[i]

    @torch.no_grad()
    def dec_grad_sq(self):
        # sum of squares of the full decoder gradients of this shard (bf16 working weights stand in for the master)
        total = torch.zeros((), device=self.norm_factor.device)
        for i in self.layers:
            s = self.norm_grad_scale(i)
            for r0, r1 in self.chunks(i):
                total += (self.dec_G[i][r0:r1].float() + s[r0:r1, None] * self.dec_W[i][r0:r1].float()).pow(2).sum()
        return total

    @torch.no_grad()
    def dec_update(self, lr, grad_scale):
        # one 8-bit adam step on every decoder, streamed through the gpu chunk by chunk: master + state in, gradient
        # (dec_G + norm term, times grad_scale for clipping) formed against the fp32 master, bitsandbytes' blockwise update,
        # master + state out, bf16 copy and row norms refreshed, gradient accumulator zeroed. pipelined over 3 streams:
        # chunk k+1 copies in and chunk k-1 copies out while chunk k updates (two gpu buffer sets, alternating)
        self.dec_step += 1
        dev = self.norm_factor.device
        qmap1, qmap2 = QMAP1.to(dev), QMAP2.to(dev)
        compute = torch.cuda.current_stream(dev)
        # [(layer, r0, r1)] every chunk of every owned decoder, in update order
        jobs = [(i, r0, r1) for i in self.layers for r0, r1 in self.chunks(i)]
        scales = {i: self.norm_grad_scale(i) for i in self.layers}  # {layer: (F,)}
        loaded = [torch.cuda.Event(), torch.cuda.Event()]  # buffer set filled by the h2d stream
        freed = [torch.cuda.Event(), torch.cuda.Event()]  # buffer set emptied by the d2h stream

        def host(k):
            # pinned cpu views of chunk k -> (master (r, M), s1 (n,), s2, absmax1 (n / QBLOCK,), absmax2)
            i, r0, r1 = jobs[k]
            M = self.dec_master[i].shape[1]
            e0, e1 = r0 * M, r1 * M
            st = self.dec_adam[i]
            return (self.dec_master[i][r0:r1], st["s1"][e0:e1], st["s2"][e0:e1],
                    st["absmax1"][e0 // QBLOCK:e1 // QBLOCK], st["absmax2"][e0 // QBLOCK:e1 // QBLOCK])

        def gpu(k):
            # gpu buffer views of chunk k, same shapes as host(k)
            i, r0, r1 = jobs[k]
            M = self.dec_master[i].shape[1]
            n = (r1 - r0) * M
            b = self.update_bufs[k % 2]
            return b["p"][:n].view(r1 - r0, M), b["s1"][:n], b["s2"][:n], b["a1"][:n // QBLOCK], b["a2"][:n // QBLOCK]

        def load(k):
            with torch.cuda.stream(self.h2d):
                self.h2d.wait_event(freed[k % 2])
                for dst, src in zip(gpu(k), host(k)):
                    dst.copy_(src, non_blocking=True)
                loaded[k % 2].record(self.h2d)

        load(0)
        for k, (i, r0, r1) in enumerate(jobs):
            if k + 1 < len(jobs):
                load(k + 1)
            compute.wait_event(loaded[k % 2])
            p, s1, s2, a1, a2 = gpu(k)
            # (r1 - r0, M) fp32
            g = (self.dec_G[i][r0:r1].float() + scales[i][r0:r1, None] * p) * grad_scale
            BF.optimizer_update_8bit_blockwise("adam", g, p, s1, s2, ADAM_BETAS[0], ADAM_BETAS[1], 0.0, 0.0, ADAM_EPS,
                                               self.dec_step, lr, qmap1, qmap2, a1, a2, 0.0, gnorm_scale=1.0,
                                               skip_zeros=False)
            self.dec_W[i][r0:r1] = p.bfloat16()
            self.dec_norm[i][r0:r1] = p.norm(dim=1)
            self.dec_G[i][r0:r1].zero_()
            updated = torch.cuda.Event()
            updated.record(compute)
            with torch.cuda.stream(self.d2h):
                self.d2h.wait_event(updated)
                for dst, src in zip(host(k), (p, s1, s2, a1, a2)):
                    dst.copy_(src, non_blocking=True)
                freed[k % 2].record(self.d2h)
        # the pinned host copies must be complete before anything reads them (next step, checkpoint)
        torch.cuda.synchronize(dev)

    def dec_state(self):
        # everything the decoders need to resume: {"step": int, i: {"master", "s1", "s2", "absmax1", "absmax2"}}
        return {"step": self.dec_step, **{i: {"master": self.dec_master[i], **self.dec_adam[i]} for i in self.layers}}

    def load_dec_state(self, state):
        self.dec_step = state["step"]
        for i in self.layers:
            self.dec_master[i].copy_(state[i]["master"])
            for k in ("s1", "s2", "absmax1", "absmax2"):
                self.dec_adam[i][k].copy_(state[i][k])
        self.refresh_decoders()

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
