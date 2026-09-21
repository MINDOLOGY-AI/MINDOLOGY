'''
GridDigest: simplified t-digest on a fixed 200-point quantile grid, 0.5% per slot.

centroid i permanently owns quantile slot [i/200, (i+1)/200] and stores the
average value of tokens that landed in it. absorb() merges a batch by
count-weighted averaging of the batch's grid values.
'''

import torch


class GridDigest:
    def __init__(self, n_neurons, n_grid=200, device="cpu"):
        self.n_grid = n_grid
        self.grid = (torch.arange(n_grid, dtype=torch.float32, device=device) + 0.5) / n_grid
        self.centroids = torch.zeros(n_grid, n_neurons, dtype=torch.float32, device=device)
        self.count = 0

    def absorb(self, x):
        # x: (..., n_neurons), any leading dims
        x = x.reshape(-1, x.shape[-1]).to(self.centroids.device, torch.float32)
        n = x.shape[0]
        q = torch.quantile(x, self.grid, dim=0)
        self.centroids = (self.centroids * self.count + q * n) / (self.count + n)
        self.count += n

    def quantile(self, p):
        # nearest grid slot, p in [0, 1]
        i = min(int(p * self.n_grid), self.n_grid - 1)
        return self.centroids[i]
