import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from typing import Optional, Tuple


class HiveMindGNN(nn.Module):
    def __init__(
        self,
        node_input_dim: int = 7,
        edge_input_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        attention_heads: int = 4
    ):
        super().__init__()
        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        self.path_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.node_encoder(node_features)
        
        for i in range(self.num_layers):
            x_new = self.convs[i](x, edge_index)
            x_new = self.norms[i](x_new)
            x_new = F.relu(x_new)
            x = x_new + x
        
        edge_logits = self._compute_edge_scores(x, edge_index)
        
        return x, edge_logits
    
    def _compute_edge_scores(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        src_emb = node_embeddings[src]
        dst_emb = node_embeddings[dst]
        
        edge_features = torch.cat([src_emb, dst_emb], dim=-1)
        scores = self.edge_mlp(edge_features)
        return scores.squeeze(-1)
    
    def predict_edge_weights(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        _, edge_logits = self.forward(node_features, edge_index, edge_attr)
        weights = torch.sigmoid(edge_logits)
        return weights
    
    def score_paths(
        self,
        node_embeddings: torch.Tensor,
        paths: torch.Tensor
    ) -> torch.Tensor:
        batch_size = paths.shape[0]
        path_length = paths.shape[1]
        
        embeddings = node_embeddings[paths]
        start_emb = embeddings[:, 0, :]
        end_emb = embeddings[:, -1, :]
        
        path_emb = torch.cat([start_emb, end_emb], dim=-1)
        scores = self.path_scorer(path_emb)
        return scores.squeeze(-1)