# src/model/hivemind_gnn/edge_scorer.py

import torch
import torch.nn as nn


class EdgeScorer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings, edge_index, edge_emb=None):
        src, dst = edge_index[0], edge_index[1]
        src_emb = node_embeddings[src]
        dst_emb = node_embeddings[dst]
        if edge_emb is not None:
            # Fuse edge embedding into source/dest embeddings
            src_emb = src_emb + edge_emb[src]
            dst_emb = dst_emb + edge_emb[dst]
        edge_features = torch.cat([src_emb, dst_emb], dim=-1)
        scores = self.mlp(edge_features)
        return scores.squeeze(-1)
