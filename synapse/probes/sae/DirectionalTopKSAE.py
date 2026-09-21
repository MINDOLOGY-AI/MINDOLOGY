import torch
import torch.nn as nn
from pathlib import Path


class DirectionalTopKSAE(nn.Module):
    def __init__(self, embed_dim, expansion_factor, k=64, direction_coeff=1.0, val_coeff=1.0):
        super().__init__()
        self.d_in = embed_dim
        self.d_sae = embed_dim * expansion_factor
        d_in, d_sae = self.d_in, self.d_sae
        self.k = k
        self.direction_coeff = direction_coeff
        self.val_coeff = val_coeff

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self._freq_counts = nn.Parameter(torch.zeros(d_sae, dtype=torch.long), requires_grad=False)
        self._fwd_count = nn.Parameter(torch.tensor(0, dtype=torch.long), requires_grad=False)
        self._recording = False

    def start_recording(self):
        self._recording = True

    def stop_recording(self):
        self._recording = False

    def _norm_decoder(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def compute_loss(self, batch_data):
        self._norm_decoder()
        # batch_data: (N, 2048)
        x = batch_data - self.b_dec
        pre = x @ self.W_enc + self.b_enc  # (N, d_sae)

        # directional score: cosine similarity between activation and each encoder direction
        dot_prod = x @ self.W_enc  # (N, d_sae) — x · w_i, no bias
        act_norm = x.norm(dim=-1, keepdim=True)  # (N, 1)
        enc_norm = self.W_enc.norm(dim=0)  # (d_sae,) — ||w_i|| per feature
        dir_score = dot_prod / (act_norm * enc_norm + 1e-8)  # (N, d_sae)

        # hybrid ranking: direction_coeff * cos_sim + val_coeff * pre_activation
        topk_score = self.direction_coeff * dir_score + self.val_coeff * pre  # (N, d_sae)

        _, topk_idx = torch.topk(topk_score, self.k, dim=-1)  # (N, k)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_idx, 1.0)
        f = pre * mask  # (N, d_sae) — k active features at raw pre-act magnitudes

        if self._recording:
            flat_idx = topk_idx.flatten().long()
            self._freq_counts.data.scatter_add_(
                0, flat_idx,
                torch.ones(flat_idx.shape[0], dtype=torch.long, device=flat_idx.device)
            )
            self._fwd_count.data += f.shape[0]

        x_hat = f @ self.W_dec + self.b_dec  # (N, d_in)
        recon_loss = ((batch_data - x_hat) ** 2).mean()
        activation_energy = batch_data.pow(2).mean()
        metrics = {
            "active": float(self.k),
            "recon_pct": (recon_loss / activation_energy * 100.0).item(),
        }
        return recon_loss, metrics

    def print_freq_dist(self, out_dir):
        total_fwd = self._fwd_count.item()
        if total_fwd == 0:
            print("  no frequency data recorded")
            return

        freq = self._freq_counts.data.float() / total_fwd * 100.0  # (d_sae,) percentages
        bins = [(i * 0.05, (i + 1) * 0.05) for i in range(60)]  # 0–3% in 0.05% steps
        bins.append((3.0, 100.0))

        lines = [f"feature frequency distribution (total activations: {total_fwd:,}, k={self.k})"]
        lines.append(f"{'bin':>16}  {'count':>8}  {'pct':>8}")
        lines.append("-" * 36)
        for lo, hi in bins:
            count = ((freq >= lo) & (freq < hi)).sum().item()
            pct = count / self.d_sae * 100.0
            label = f"{lo:.2f}%-{hi:.2f}%" if hi < 100 else f"{lo:.2f}%+"
            lines.append(f"{label:>16}  {count:>8,}  {pct:>7.2f}%")

        text = "\n".join(lines)
        print(text)
        out_dir = Path(out_dir)
        (out_dir / "freq_dist.txt").write_text(text)
