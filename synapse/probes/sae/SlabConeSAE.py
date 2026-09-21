import torch
import torch.nn as nn

# =============================================================================
# SLAB-CONE SAE
# =============================================================================
# Each feature fires only when the input passes THREE hard gates at once:
#
#   slab lower gate:  pre-activation pi must be ABOVE a learnable threshold
#   slab upper gate:  pre-activation pi must be BELOW a learnable threshold
#   cone gate:        angle between input and encoder direction must be
#                     BELOW a learnable threshold
#
# The slab forces magnitude specialization (a feature owns a band of
# activation strengths, not "everything above threshold") and the cone forces
# direction specialization (a feature owns a cone around its encoder row).
# A feature's receptive field is the intersection: a truncated cone segment.
#
# -----------------------------------------------------------------------------
# BACKPROP DESIGN (JumpReLU's calibrated STE, generalized to any gate)
# -----------------------------------------------------------------------------
# Every gate is  gate = H(+/-(s - theta))  where s is the gate's score
# (pi for the slabs, angle for the cone) and theta is its learnable threshold.
# The hard gate has zero true derivative w.r.t. theta, so we use a
# straight-through estimator whose batch average equals the gradient of the
# EXPECTED loss (full derivation in jumprelu_complete.md):
#
#   grad_theta = sign * (1/eps) * rect((s - theta)/eps) * grad_output
#
#     sign = -1 for LOWER bounds: raising theta kills boundary features
#            (reconstruction worsens, sparsity improves)
#     sign = +1 for UPPER bounds: raising theta revives boundary features
#            (reconstruction improves, sparsity worsens)
#
#   grad_score = None  -->  BLOCKED, ALWAYS.
#     Neither the reconstruction loss nor the sparsity penalty may move the
#     encoder through a gate. If sparsity could flow into the scores, the
#     encoder would dodge the penalty by shrinking pre-activations, inflating
#     them past the upper slab, or rotating directions out of the cone.
#     Reconstruction reaches the encoder ONLY through the explicit magnitude
#     path (features = pi * gates). The encoder learns to reconstruct;
#     the thetas alone handle the sparsity tradeoff.
#
# WHY THE COEFFICIENT IS +/-1/eps AND NOT -theta/eps:
# In JumpReluSAE the STE node IS the magnitude product (f = z * H), so
# grad_output arrives without the magnitude and the -theta/eps coefficient is
# needed to cancel the 1/theta hidden in the reconstruction chain rule.
# Here each gate is a SEPARATE multiplicative node in
# features = pi * g_lower * g_upper * g_cone, so the product rule makes
# grad_output already carry pi AND the other gates' values. The correct
# coefficient is then just +/-1/eps. This is also what makes the cone gate
# work: a cone-boundary sample's firing magnitude is its own pi (decoupled
# from the angle), and the product structure supplies it automatically.
# =============================================================================


class GateSTE(torch.autograd.Function):
    # one hard threshold gate with calibrated straight-through backprop
    # forward: 1 if the score passes the threshold, else 0
    # backward: theta gets the rect-window pseudo gradient, score gets NOTHING
    @staticmethod
    def forward(ctx, score, theta, eps, is_lower):
        # score    : (B, M) per-sample gate input (pre-activation or angle)
        # theta    : (M,)   learnable threshold
        # eps      : scalar rect window width (KDE bandwidth)
        # is_lower : True for lower bounds (fire if score > theta)
        ctx.save_for_backward(score, theta)
        ctx.eps = eps
        ctx.is_lower = is_lower
        if is_lower:
            gate = score > theta
        else:
            gate = score < theta
        return gate.to(score.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        score, theta = ctx.saved_tensors

        # rect((s - theta)/eps): 1 only for samples sitting at this boundary
        in_window = ((score - theta).abs() < ctx.eps / 2).to(grad_output.dtype)

        # sign convention from the header:
        #   lower bound: raising theta kills boundary features  -> -1
        #   upper bound: raising theta revives boundary features -> +1
        sign = -1.0 if ctx.is_lower else 1.0

        # pseudo gradient for theta; sum over the batch -> (M,)
        grad_theta = sign * (in_window / ctx.eps) * grad_output
        grad_theta = grad_theta.sum(dim=0)

        # score grad = None -> BLOCKED. gates never teach the encoder.
        return None, grad_theta, None, None


class SlabConeSAE(nn.Module):
    def __init__(self, embed_dim, expansion_factor, sparsity_coeff=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = embed_dim * expansion_factor
        self.sparsity_coeff = sparsity_coeff
        self.band_eps = 0.05  # rect window width (KDE bandwidth)

        # dictionary weights (trained by RECONSTRUCTION only, via the
        # magnitude path)
        self.encoder_weight = nn.Parameter(torch.randn(self.feature_dim, self.embed_dim) / self.embed_dim ** 0.5)  # (out, in) format
        self.encoder_bias = nn.Parameter(torch.randn(self.feature_dim) / self.embed_dim ** 0.5)
        self.decoder_weight = nn.Parameter(torch.randn(self.embed_dim, self.feature_dim) / self.feature_dim ** 0.5)
        self.decoder_bias = nn.Parameter(torch.randn(self.embed_dim) / self.embed_dim ** 0.5)

        # gate thresholds (trained by BOTH reconstruction and sparsity, via
        # GateSTE). start in the range where pre-activations actually live
        self.slab_gate_lower_bound = nn.Parameter(torch.full((self.feature_dim,), 0.1))
        self.slab_gate_upper_bound = nn.Parameter(torch.full((self.feature_dim,), 0.3))
        self.cone_gate_upper_bound = nn.Parameter(torch.full((self.feature_dim,), 1.57))  # ~pi/2, a right angle

    def encode(self, input):
        input = input.to(self.encoder_weight.dtype)

        # one projection, reused by both the slab scores and the cone angles
        centered = input - self.decoder_bias                         # (B, embed_dim)
        projection = centered @ self.encoder_weight.T                # (B, feature_dim)
        pre_gate_features = projection + self.encoder_bias           # (B, feature_dim)

        # angle between the centered input and each encoder row
        cos_sim = projection / (centered.norm(dim=-1, keepdim=True) * self.encoder_weight.norm(dim=-1))
        angles = torch.acos(cos_sim.clamp(-1 + 1e-7, 1 - 1e-7))      # (B, feature_dim)

        # three hard gates; each theta learns via GateSTE, scores stay blocked
        slab_lower_gate = GateSTE.apply(pre_gate_features, self.slab_gate_lower_bound, self.band_eps, True)
        slab_upper_gate = GateSTE.apply(pre_gate_features, self.slab_gate_upper_bound, self.band_eps, False)
        cone_gate = GateSTE.apply(angles, self.cone_gate_upper_bound, self.band_eps, False)

        # all three gates must be on for a feature to fire
        combined_gate = slab_lower_gate * slab_upper_gate * cone_gate

        # magnitude path: the pre-activation passes through untouched when the
        # feature fires. this is the ONLY route reconstruction gradients have
        # back to the encoder.
        features = pre_gate_features * combined_gate
        return features, combined_gate

    def decode(self, features):
        return features @ self.decoder_weight.T

    def _norm_decoder(self):
        with torch.no_grad():
            self.decoder_weight.data /= self.decoder_weight.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

    def compute_loss(self, batch_data):
        self._norm_decoder()
        features, combined_gate = self.encode(batch_data)

        active_count = combined_gate.sum(-1).mean()  # scalar
        sparsity_loss = self.sparsity_coeff * active_count

        pred = self.decode(features)
        recon_loss = (pred - batch_data).pow(2).mean()
        activation_energy = batch_data.pow(2).mean()
        recon_pct = (recon_loss / activation_energy) * 100.0
        total = sparsity_loss + recon_loss

        metrics = {
            "active": active_count.item(),
            "recon_pct": recon_pct.item(),
            "avg_lower": self.slab_gate_lower_bound.mean().item(),
            "avg_upper": self.slab_gate_upper_bound.mean().item(),
            "avg_cone": self.cone_gate_upper_bound.mean().item(),
        }
        return total, metrics
