# src/model/hivemind_gnn/gcn_stack.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNStack(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_heads = attention_heads
        self.head_dim = hidden_dim // attention_heads

        self.convs = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        # Multi-head attention over node features (DIFFGAT-style)
        self.attn_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.edge_update = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, edge_emb=None):
        for i, (conv, norm, proj) in enumerate(
            zip(self.convs, self.norms, self.attn_projs)
        ):
            # GCN message passing
            x_new = conv(x, edge_index)
            # Multi-head feature-difference attention (DIFFGAT 2025)
            B = x_new.shape[0]
            x_proj = proj(x_new).view(B, self.attention_heads, self.head_dim)
            # Compute attention weights: (B, attention_heads)
            attn_scores = (x_proj * x_proj.mean(dim=0, keepdim=True)).sum(dim=-1)
            diff_weight = F.softmax(attn_scores, dim=-1)
            # Apply attention and reshape back to (B, hidden_dim)
            x_new_heads = x_new.view(B, self.attention_heads, self.head_dim)
            x_attended = (x_new_heads * diff_weight.unsqueeze(-1)).view(B, self.hidden_dim)
            x_new = norm(x_attended)
            x_new = F.relu(x_new)
            x = x + x_new
            # Update edge embeddings if provided (EdgeGFL / Rohatgi 2025)
            if edge_emb is not None:
                src, dst = edge_index[0], edge_index[1]
                edge_emb = self.edge_update[i](
                    torch.cat([edge_emb[src], edge_emb[dst]], dim=-1)
                )
        return x, edge_emb
