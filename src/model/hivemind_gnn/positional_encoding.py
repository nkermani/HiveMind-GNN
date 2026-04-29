# src/model/hivemind_gnn/positional_encoding.py

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        rw_steps: int = 16,
    ):
        super().__init__()
        self.rw_steps = rw_steps
        # Random-walk positional encoding (RWSE, benchmarked 2025)
        self.rw_mlp = nn.Sequential(
            nn.Linear(rw_steps, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_index, num_nodes, device):
        row, col = edge_index
        adj = torch.zeros(
            num_nodes, num_nodes, device=device
        )
        adj[row, col] = 1.0
        rw = adj.clone()
        rw_feats = [rw.sum(dim=-1)]
        for _ in range(1, self.rw_steps):
            rw = rw @ adj
            rw_feats.append(rw.sum(dim=-1))
        rw_feats = torch.stack(rw_feats, dim=-1)
        return self.rw_mlp(rw_feats)
