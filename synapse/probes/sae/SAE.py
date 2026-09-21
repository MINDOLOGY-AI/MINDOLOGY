# normal SAE

import torch
import torch.nn as nn

class SAE (nn.Module):
    def __init__ (self, embed_dim, expansion_factor, sparsity_coeff=1e-3):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = embed_dim * expansion_factor
        self.sparsity_coeff = sparsity_coeff

        self.encoder_weight = nn.Parameter(torch.randn(self.feature_dim, self.embed_dim) / self.embed_dim ** 0.5) # (out, in) format
        self.encoder_bias = nn.Parameter(torch.randn(self.feature_dim) / self.embed_dim ** 0.5)
        self.decoder_weight = nn.Parameter(torch.randn(self.embed_dim, self.feature_dim) / self.feature_dim ** 0.5)
        # find a center to substract from all activations
        self.decoder_bias = nn.Parameter(torch.randn(self.embed_dim) / self.embed_dim ** 0.5)

        self.relu = nn.ReLU()

    def encode(self, input):
        pre_gate_features = (input - self.decoder_bias) @ self.encoder_weight.T + self.encoder_bias
        features = self.relu(pre_gate_features)
        return features

    def decode(self, features):
        return features @ self.decoder_weight.T

    def compute_loss(self, batch_data):
        features = self.encode(batch_data)
        sparsity_loss = features.mean() * self.sparsity_coeff
        pred = self.decode(features)
        recon_loss = (batch_data - pred).pow(2).mean()
        loss = sparsity_loss + recon_loss
        activation_energy = batch_data.pow(2).mean()
        metrics = {
            "l1": features.mean().item(),
            "active": (features > 0).float().sum(dim=-1).mean().item(),
            "recon_pct": (recon_loss / activation_energy * 100.0).item(),
        }
        return loss, metrics
