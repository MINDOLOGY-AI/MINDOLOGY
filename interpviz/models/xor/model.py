# model.py -- xor demo model for interpviz

import torch
import torch.nn as nn


class XORNet(nn.Module):
    """2-layer feedforward network for XOR."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.out = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (batch, 2) -> (batch, 8)
        x = self.hidden(x)
        x = self.relu(x)
        # (batch, 8) -> (batch, 1)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
