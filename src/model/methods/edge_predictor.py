# src/model/predictor/edge_predictor.py

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..hivemind_gnn import HiveMindGNN


class EdgePredictor(nn.Module):
    def __init__(
        self,
        gnn: Optional[HiveMindGNN] = None,
        hidden_dim: int = 64,
        node_input_dim: int = 7,
        edge_input_dim: int = 2,
        num_layers: int = 3
    ):
        super().__init__()
        self.gnn = gnn or HiveMindGNN(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embeddings, edge_logits = self.gnn(node_features, edge_index, edge_attr)

        loss = None
        if edge_labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                edge_logits, edge_labels.float()
            )

        edge_probs = torch.sigmoid(edge_logits)
        return edge_probs, loss

    def predict_optimal_edges(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        top_k: int = 10
    ) -> torch.Tensor:
        edge_probs, _ = self.forward(node_features, edge_index, edge_attr)

        topk_values, topk_indices = torch.topk(edge_probs, min(top_k, len(edge_probs)))

        mask = torch.zeros_like(edge_probs, dtype=torch.bool)
        mask[topk_indices] = True

        return edge_probs * mask.float()
