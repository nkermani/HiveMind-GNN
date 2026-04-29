# src/model/hivemind_gnn/hivemind_gnn.py

import torch
from torch.nn import Module
from typing import Optional, Tuple

from .encoders import NodeEncoder, EdgeEncoder
from .gcn_stack import GCNStack
from .edge_scorer import EdgeScorer
from .path_scorer import PathScorer
from .positional_encoding import PositionalEncoding


class HiveMindGNN(Module):
    def __init__(
        self,
        node_input_dim: int = 7,
        edge_input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        attention_heads: int = 4,
    ):
        super().__init__()
        self.node_encoder = NodeEncoder(node_input_dim, hidden_dim, dropout)
        self.edge_encoder = EdgeEncoder(edge_input_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, num_layers)
        self.gcn_stack = GCNStack(hidden_dim, num_layers, attention_heads, dropout)
        self.edge_scorer = EdgeScorer(hidden_dim, dropout)
        self.path_scorer = PathScorer(hidden_dim, dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.node_encoder(node_features)
        edge_emb = self.edge_encoder(edge_attr)
        pos = self.pos_enc(edge_index, x.shape[0], x.device)
        x = x + pos
        x, edge_emb = self.gcn_stack(x, edge_index, edge_emb)
        edge_logits = self.edge_scorer(x, edge_index, edge_emb)
        return x, edge_logits

    def predict_edge_weights(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        _, edge_logits = self.forward(node_features, edge_index, edge_attr)
        return torch.sigmoid(edge_logits)

    def score_paths(
        self,
        node_embeddings: torch.Tensor,
        paths: torch.Tensor
    ) -> torch.Tensor:
        return self.path_scorer(node_embeddings, paths)
