# src/model/hivemind_gnn/path_scorer.py

import torch.nn as nn


class PathScorer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, node_embeddings, paths):
        embeddings = node_embeddings[paths]
        start_emb = embeddings[:, 0, :]
        end_emb = embeddings[:, -1, :]
        path_emb = torch.cat([start_emb, end_emb], dim=-1)
        scores = self.scorer(path_emb)
        return scores.squeeze(-1)
