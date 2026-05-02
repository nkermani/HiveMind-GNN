import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
from src.env import FlowerFieldGenerator, HiveMindEnvironment
from src.model import EdgePredictor
from src.train import Trainer, GraphDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch

from scripts.visualization.methods.graph_viz import visualize_graph
from scripts.visualization.methods.features import visualize_feature_distributions
from scripts.visualization.methods.bees import visualize_bee_navigation
from scripts.visualization.methods.predictions import visualize_gnn_predictions
from scripts.visualization.methods.training import visualize_training_progress
from scripts.visualization.methods.batch import visualize_batch_of_graphs

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

def run_visualization_demo():
    print("=" * 60)
    print("HiveMind-GNN Visualization Demo")
    print("=" * 60)

    os.makedirs('visualizations', exist_ok=True)

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
    model, env, graph = run_visualization_demo()
