import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from src.env import FlowerFieldGenerator
import torch

def visualize_gnn_predictions(graph, model):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    generator = FlowerFieldGenerator(num_nodes=graph.number_of_nodes(), seed=42)
    obs = {
        'node_features': np.array([generator.get_feature_vector(i, graph) for i in range(graph.number_of_nodes())], dtype=np.float32),
        'edge_index': np.array(list(graph.edges())).T,
        'edge_attr': np.array([[generator.get_edge_weight(u, v, graph), 0] for u, v in graph.edges()], dtype=np.float32),
    }

    node_features = torch.tensor(obs['node_features'])
    edge_index = torch.tensor(obs['edge_index'])
    edge_attr = torch.tensor(obs['edge_attr'])

    model.eval()
    with torch.no_grad():
        edge_probs, _ = model(node_features, edge_index, edge_attr)
        edge_probs = edge_probs.cpu().numpy()

    sorted_probs = np.sort(edge_probs)[::-1]
    axes[0].bar(range(len(sorted_probs)), sorted_probs, color='#6C5CE7', alpha=0.7)
    axes[0].axhline(y=np.mean(edge_probs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(edge_probs):.3f}')
    axes[0].set_xlabel('Edge Index (sorted by probability)')
    axes[0].set_ylabel('Probability of Being Optimal')
    axes[0].set_title('GNN Edge Probability Predictions', fontsize=12, fontweight='bold')
    axes[0].legend()

    pos = nx.spring_layout(graph, seed=42, k=2)
    edge_colors = [plt.cm.RdYlGn(prob) for prob in edge_probs]
    edge_widths = [1 + prob * 3 for prob in edge_probs]

    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=100, alpha=0.8, ax=axes[1])
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=edge_widths, alpha=0.8, ax=axes[1])
    axes[1].set_title('Graph Colored by Edge Probability', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], shrink=0.5)
    cbar.set_label('Edge Probability')

    plt.tight_layout()
    return fig
