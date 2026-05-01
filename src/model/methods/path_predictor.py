# src/model/predictor/path_predictor.py

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..hivemind_gnn import HiveMindGNN


class PathPredictor(nn.Module):
    def __init__(self, gnn: Optional[HiveMindGNN] = None):
        super().__init__()
        self.gnn = gnn

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        paths: torch.Tensor
    ) -> torch.Tensor:
        embeddings, _ = self.gnn(node_features, edge_index, edge_attr)
        path_scores = self.gnn.score_paths(embeddings, paths)
        return path_scores

    def select_best_paths(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        candidate_paths: list,
        top_k: int = 3
    ) -> list:
        if not candidate_paths:
            return []

        max_len = max(len(p) for p in candidate_paths)
        padded_paths = []
        for p in candidate_paths:
            padded = p + [p[-1]] * (max_len - len(p))
            padded_paths.append(padded)

        paths_tensor = torch.tensor(padded_paths, dtype=torch.long)

        scores = self.forward(node_features, edge_index, edge_attr, paths_tensor)

        _, top_indices = torch.topk(scores, min(top_k, len(scores)))

        return [candidate_paths[i] for i in top_indices.tolist()]
