import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from collections import deque
import random

from src.env import FlowerFieldGenerator, HiveMindEnvironment
from src.model import HiveMindGNN, EdgePredictor
from src.train import Trainer, GraphDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch

plt.style.use('seaborn-v0_8-whitegrid')


def visualize_graph(graph, title="Flower Field Graph"):
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.spring_layout(graph, seed=42, k=2)

    node_colors = []
    node_sizes = []
    for i in range(graph.number_of_nodes()):
        if graph.nodes[i].get('is_source', False):
            node_colors.append('#FFD700')
            node_sizes.append(400)
        elif graph.nodes[i].get('is_sink', False):
            node_colors.append('#FF6B6B')
            node_sizes.append(400)
        else:
            nectar = graph.nodes[i].get('nectar_density', 0)
            node_colors.append(plt.cm.YlOrRd(nectar))
            node_sizes.append(150 + nectar * 200)

    edge_colors = []
    edge_widths = []
    for u, v in graph.edges():
        weight = graph.edges[u, v].get('base_weight', 1.0)
        edge_colors.append('#888888')
        edge_widths.append(0.5 + (1 / weight) * 0.5)

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=edge_widths, alpha=0.4, arrows=True, ax=ax)

    labels = {i: f'{i}' for i in range(graph.number_of_nodes()) if graph.nodes[i].get('is_source', False) or graph.nodes[i].get('is_sink', False)}
    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_weight='bold', ax=ax)

    source_patch = mpatches.Patch(color='#FFD700', label='Sources (Hive)')
    sink_patch = mpatches.Patch(color='#FF6B6B', label='Sinks (Flowers)')
    nectar_patch = mpatches.Patch(color='red', label='High Nectar', alpha=0.5)
    empty_patch = mpatches.Patch(color='wheat', label='Low Nectar', alpha=0.5)
    ax.legend(handles=[source_patch, sink_patch, nectar_patch, empty_patch], loc='upper left')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig, ax


def visualize_feature_distributions(graph, generator):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    nectar_levels = [graph.nodes[i].get('nectar_density', 0) for i in range(graph.number_of_nodes())]
    occupancy = [graph.nodes[i].get('current_occupancy', 0) for i in range(graph.number_of_nodes())]
    out_degree = [graph.out_degree(i) for i in range(graph.number_of_nodes())]
    in_degree = [graph.in_degree(i) for i in range(graph.number_of_nodes())]

    axes[0, 0].hist(nectar_levels, bins=20, color='#FFD700', edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('Nectar Density Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Nectar Level')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].hist(out_degree, bins=15, color='#4ECDC4', edgecolor='black', alpha=0.8)
    axes[0, 1].set_title('Out-Degree Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Out-Degree')
    axes[0, 1].set_ylabel('Count')

    axes[1, 0].hist(in_degree, bins=15, color='#FF6B6B', edgecolor='black', alpha=0.8)
    axes[1, 0].set_title('In-Degree Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('In-Degree')
    axes[1, 0].set_ylabel('Count')

    edge_weights = [generator.get_edge_weight(u, v, graph) for u, v in graph.edges()]
    axes[1, 1].hist(edge_weights, bins=30, color='#95E1D3', edgecolor='black', alpha=0.8)
    axes[1, 1].set_title('Edge Weight Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Edge Weight (lower = better)')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    return fig


def visualize_bee_navigation(graph, env, max_steps=20):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    pos = nx.spring_layout(graph, seed=42, k=2)

    node_colors = []
    for i in range(graph.number_of_nodes()):
        if graph.nodes[i].get('is_source', False):
            node_colors.append('#FFD700')
        elif graph.nodes[i].get('is_sink', False):
            node_colors.append('#FF6B6B')
        else:
            node_colors.append('#A8E6CF')

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=200, alpha=0.7, ax=axes[0])
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.3, arrows=True, ax=axes[0])
    axes[0].set_title('Initial State', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    env.reset(graph)

    bee_positions = [bee.current_node for bee in env.bees]
    nx.draw_networkx_nodes(graph, pos, nodelist=bee_positions, node_color='blue', node_size=300, alpha=0.9, ax=axes[1])

    for idx, bee in enumerate(env.bees):
        x, y = pos[bee.current_node]
        axes[1].annotate(f'B{bee.bee_id}', (x, y), fontsize=8, ha='center', va='bottom', color='white', fontweight='bold')

    axes[1].set_title('Bees Placed at Sources', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    return fig


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


def visualize_training_progress(history, val_losses=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    train_loss = history['train_loss']
    val_loss = val_losses if val_losses else history.get('val_loss', [])

    axes[0].plot(train_loss, 'b-', linewidth=2, alpha=0.7, label='Training Loss')
    if val_loss:
        axes[0].plot(val_loss, 'r-', linewidth=2, alpha=0.7, label='Validation Loss')

    if len(train_loss) > 10:
        window = min(5, len(train_loss) // 5)
        smoothed = np.convolve(train_loss, np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(train_loss)), smoothed, 'b--', linewidth=2, label=f'Smoothed (window={window})')

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('BCE Loss', fontsize=12)
    axes[0].set_title('Loss Over Training', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(['Initial', 'Final'], [train_loss[0], train_loss[-1]], color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    if val_loss:
        axes[1].bar(['Val Initial', 'Val Final'], [val_loss[0], val_loss[-1]], color=['#FF9999', '#99EECB'], edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('BCE Loss', fontsize=12)
    axes[1].set_title('Loss Comparison: Start vs End', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    for i, v in enumerate([train_loss[0], train_loss[-1]]):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return fig


def visualize_batch_of_graphs(generator):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx in range(6):
        gen = FlowerFieldGenerator(num_nodes=30, seed=idx * 10)
        g = gen.generate()

        pos = nx.spring_layout(g, seed=42, k=1.5)

        node_colors = []
        for i in range(g.number_of_nodes()):
            if g.nodes[i].get('is_source', False):
                node_colors.append('#FFD700')
            elif g.nodes[i].get('is_sink', False):
                node_colors.append('#FF6B6B')
            else:
                nectar = g.nodes[i].get('nectar_density', 0)
                node_colors.append(plt.cm.YlOrRd(nectar))

        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=80, alpha=0.8, ax=axes[idx])
        nx.draw_networkx_edges(g, pos, edge_color='gray', alpha=0.3, ax=axes[idx])
        axes[idx].set_title(f'Flower Field #{idx + 1}\n({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)', fontsize=10)
        axes[idx].axis('off')

    plt.tight_layout()
    return fig


def run_visualization_demo():
    print("=" * 60)
    print("HiveMind-GNN Visualization Demo")
    print("=" * 60)

    generator = FlowerFieldGenerator(num_nodes=40, seed=42)
    graph = generator.generate()

    print("\n[1/6] Generating flower field graph...")
    fig1, ax1 = visualize_graph(graph)
    fig1.savefig('visualizations/01_flower_field_graph.png', dpi=150, bbox_inches='tight')
    print("   Saved: visualizations/01_flower_field_graph.png")

    print("\n[2/6] Analyzing feature distributions...")
    fig2 = visualize_feature_distributions(graph, generator)
    fig2.savefig('visualizations/02_feature_distributions.png', dpi=150, bbox_inches='tight')
    print("   Saved: visualizations/02_feature_distributions.png")

    print("\n[3/6] Initializing environment with bees...")
    env = HiveMindEnvironment(num_bees=8, graph_generator=generator)
    fig3 = visualize_bee_navigation(graph, env)
    fig3.savefig('visualizations/03_bee_initialization.png', dpi=150, bbox_inches='tight')
    print("   Saved: visualizations/03_bee_initialization.png")

    print("\n[4/6] Training GNN for 50 epochs...")
    dataset = GraphDataset(generator, num_samples=200)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = PyGDataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=16)

    model = EdgePredictor()
    trainer = Trainer(model, learning_rate=1e-3)

    for epoch in range(50):
        loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        trainer.train_losses.append(loss)
        trainer.val_losses.append(val_loss)
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")

    fig4 = visualize_training_progress({'train_loss': trainer.train_losses}, trainer.val_losses)
    fig4.savefig('visualizations/04_training_progress.png', dpi=150, bbox_inches='tight')
    print("   Saved: visualizations/04_training_progress.png")

    print("\n[5/6] Visualizing GNN predictions...")
    fig5 = visualize_gnn_predictions(graph, model)
    fig5.savefig('visualizations/05_gnn_predictions.png', dpi=150, bbox_inches='tight')
    print("   Saved: visualizations/05_gnn_predictions.png")

    print("\n[6/6] Generating diverse graph samples...")
    fig6 = visualize_batch_of_graphs(generator)
    fig6.savefig('visualizations/06_graph_samples.png', dpi=150, bbox_inches='tight')
    print("   Saved: visualizations/06_graph_samples.png")

    print("\n" + "=" * 60)
    print("All visualizations saved to 'visualizations/' directory")
    print("=" * 60)

    return model, env, graph


if __name__ == '__main__':
    import os
    os.makedirs('visualizations', exist_ok=True)
    model, env, graph = run_visualization_demo()