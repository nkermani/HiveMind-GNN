# src/model/hivemind_gnn/encoders.py

import torch.nn as nn


class EdgeEncoder(nn.Module):
    def __init__(
        self,
        edge_input_dim: int = 2,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, edge_attr):
        return self.layers(edge_attr)


class NodeEncoder(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 7,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)
