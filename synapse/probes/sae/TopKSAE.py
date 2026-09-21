import torch
import torch.nn as nn
# =============================================================================
# TopK SAE — built-in sparsity via top-k selection, no penalty coefficient
# =============================================================================
class TopKSAE(nn.Module):
    def __init__(self, embed_dim, expansion_factor, k=64):
        super().__init__()
        self.d_in = embed_dim
        self.d_sae = embed_dim * expansion_factor
        d_in, d_sae = self.d_in, self.d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def _norm_decoder(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def compute_loss(self, batch_data):
        self._norm_decoder()
        # batch_data: (B*seq_len, 2048)
        x = batch_data - self.b_dec
        pre = x @ self.W_enc + self.b_enc             # (N, d_sae)
        _, topk_idx = torch.topk(pre, self.k, dim=-1)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_idx, 1.0)
        f = pre * mask                                 # (N, d_sae) — k active features
        x_hat = f @ self.W_dec + self.b_dec            # (N, d_in)
        recon_loss = ((batch_data - x_hat) ** 2).mean()
        activation_energy = batch_data.pow(2).mean()
        active_count = (f > 0).float().sum(dim=-1).mean()
        metrics = {
            "active": active_count.item(),
            "recon_pct": (recon_loss / activation_energy * 100.0).item(),
        }
        return recon_loss, metrics
