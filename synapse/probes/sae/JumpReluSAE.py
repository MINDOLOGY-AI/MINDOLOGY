import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. JUMPRELU STE — used in the forward pass to produce feature magnitudes
# =============================================================================
# Forward:  out = z * (z > theta)   ... hard threshold, no gradient naturally
# Backward: we LIE to autograd and inject a fake gradient for theta only
class JumpReLU_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, theta, eps=0.001):
        # z      : pre-activations  shape (B, M)
        # theta  : thresholds       shape (M,)
        # eps    : KDE bandwidth    scalar
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        mask = (z > theta).float()      # hard Heaviside gate
        return z * mask                  # identity above theta, zero below

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output comes from dL_recon / d(out)
        z, theta = ctx.saved_tensors
        eps = ctx.eps

        # --- TRUE gradient w.r.t. z (reconstruction flows normally) ---
        # If z > theta, JumpReLU is identity, so grad passes straight through.
        # If z < theta, output is zero, so grad is zero.
        grad_z = (z > theta).float() * grad_output   # (B, M)

        # --- PSEUDO-GRADIENT w.r.t. theta (STE / KDE trick) ---
        # The real derivative is zero everywhere (flat function).
        # We replace it with a narrow rectangle window around the threshold.
        # This approximates the gradient of the EXPECTED loss.
        in_window = ((z - theta).abs() < eps / 2).float()   # (B, M)
        # The paper's Eq. (11):  -(theta/eps) * rect((z-theta)/eps) * grad
        grad_theta = -(theta / eps) * in_window * grad_output  # (B, M)
        grad_theta = grad_theta.sum(dim=0)                   # (M,)  sum over batch

        # KEY: grad_z is NON-ZERO  -> reconstruction loss can reach W_enc
        #      grad_theta is NON-ZERO -> theta learns from reconstruction
        return grad_z, grad_theta, None


# =============================================================================
# 2. HEAVISIDE STE — used ONLY for the L0 sparsity penalty
# =============================================================================
# Forward:  out = 1.0 if z > theta else 0.0
# Backward: we zero grad w.r.t. z, but give a fake grad w.r.t. theta
class Heaviside_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, theta, eps=0.001):
        ctx.save_for_backward(z, theta)
        ctx.eps = eps
        return (z > theta).float()       # hard 0/1

    @staticmethod
    def backward(ctx, grad_output):
        z, theta = ctx.saved_tensors
        eps = ctx.eps

        # --- CRITICAL: we explicitly ZERO the gradient w.r.t. z ---
        # This is the whole trick. The sparsity loss is a "dead end".
        # W_enc will NEVER see this gradient because grad_z = 0.
        grad_z = torch.zeros_like(z)     # (B, M)  <-- SPARSITY BLOCKED

        # --- PSEUDO-GRADIENT w.r.t. theta (STE for L0 penalty) ---
        # Paper's Eq. (12):  -(1/eps) * rect((z-theta)/eps) * grad
        in_window = ((z - theta).abs() < eps / 2).float()   # (B, M)
        grad_theta = -(1.0 / eps) * in_window * grad_output  # (B, M)
        grad_theta = grad_theta.sum(dim=0)                   # (M,)

        # KEY: grad_z is ZERO    -> sparsity loss CANNOT reach W_enc
        #      grad_theta is NON-ZERO -> theta learns from sparsity penalty
        return grad_z, grad_theta, None


# =============================================================================
# 3. JUMPRELU SAE MODULE
# =============================================================================
class JumpReLU_SAE(nn.Module):
    def __init__(self, embed_dim, expansion_factor, sparsity_coeff=0.1, eps=0.05):
        super().__init__()
        self.d_in = embed_dim
        self.d_sae = embed_dim * expansion_factor
        d_in, d_sae = self.d_in, self.d_sae
        self.sparsity_coeff = sparsity_coeff
        self.eps = eps

        # Normal encoder/decoder weights (trained by RECONSTRUCTION only)
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Threshold theta — trained by BOTH reconstruction and sparsity
        # Stored as log(theta) to keep it positive (paper's trick)
        self.log_theta = nn.Parameter(torch.ones(d_sae) * (-2.3))  # theta ≈ 0.1

    def _norm_decoder(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def forward(self, x):
        # x: (B, d_in)  LM activation vectors
        # --- Pre-encoder bias (optional, from Bricken et al.) ---
        x = x - self.b_dec

        # --- Encoder: pre-activations ---
        # pi(x) = W_enc^T x + b_enc   shape (B, M)
        pre = F.relu(x @ self.W_enc + self.b_enc)   # ReLU here is a safety
                                                    # to stop negative pre-acts
                                                    # from biasing the STE

        # --- Threshold ---
        theta = torch.exp(self.log_theta)           # (M,)  positive

        # --- JumpReLU: feature magnitudes ---
        # This is the gate that decides WHICH features fire AND their magnitude.
        # Reconstruction loss backprops through here normally to W_enc.
        f = JumpReLU_STE.apply(pre, theta, self.eps)  # (B, M)

        # --- Decoder ---
        x_hat = f @ self.W_dec + self.b_dec          # (B, d_in)

        return x_hat, f, pre, theta

    def compute_loss(self, batch_data):
        self._norm_decoder()
        x_hat, f, pre, theta = self(batch_data)
        # L_recon = ||x - x_hat||^2 per-element
        recon_loss = ((batch_data - x_hat) ** 2).mean()
        # L_sparse = coeff × L0
        l0_per_example = Heaviside_STE.apply(pre, theta, self.eps).sum(dim=-1)  # (B,)
        sparsity_loss = self.sparsity_coeff * l0_per_example.mean()
        total = recon_loss + sparsity_loss
        metrics = {
            "l0": l0_per_example.mean().item(),
            "recon_pct": (recon_loss / batch_data.pow(2).mean() * 100.0).item(),
            "avg_theta": theta.mean().item(),
        }
        return total, metrics


# =============================================================================
# 4. MINIMAL DEMO
# =============================================================================
if __name__ == "__main__":
    B = 4096
    model = JumpReLU_SAE(768, 16, eps=0.001)
    opt = torch.optim.Adam(model.parameters(), lr=7e-5, betas=(0.0, 0.999))

    # Fake activations at a realistic scale (pre-acts must reach theta ≈ 0.1
    # for anything to fire; unit-variance inputs do that with 0.01-scaled weights)
    x = torch.randn(B, model.d_in)

    loss, metrics = model.compute_loss(x)
    print(f"Total: {loss.item():.4f} | Recon%: {metrics['recon_pct']:.4f} | L0: {metrics['l0']:.4f} | Avg theta: {metrics['avg_theta']:.2f}")

    # Verify gradient flow
    loss.backward()
    print(f"W_enc grad norm:  {model.W_enc.grad.norm().item():.6f}")
    print(f"log_theta grad:   {model.log_theta.grad.abs().mean().item():.6f}")
    # If you inspect model.W_enc.grad, you'll see it ONLY came from recon.
    # model.log_theta.grad is a sum of two contributions.
