from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import random_split; import torch
from src.env import FlowerFieldGenerator, HiveMindEnvironment
from src.model import EdgePredictor
from src.training import GraphDataset, Trainer
from src.training.plotting import plot_training_history, plot_edge_predictions
from torch_geometric.data import Data

def main():
    print("=" * 60 + "\nHiveMind-GNN Training\n" + "=" * 60)
    g = FlowerFieldGenerator(num_nodes=50, num_sources=5, num_sinks=5, density=0.3, seed=42)
    ds = GraphDataset(g, num_samples=500)
    ts = int(0.8 * len(ds))
    td, vd = random_split(ds, [ts, len(ds) - ts], generator=torch.Generator().manual_seed(42))
    tl = PyGDataLoader(td, batch_size=32, shuffle=True)
    vl = PyGDataLoader(vd, batch_size=32, shuffle=False)
    print(f"Train: {len(td)}\nVal: {len(vd)}\n")
    m = EdgePredictor()
    t = Trainer(m, learning_rate=1e-3)
    print(f"Device: {t.device}\nParams: {sum(p.numel() for p in m.parameters()):,}\n")
    print("Starting training...")
    h = t.train(tl, vl, num_epochs=100, checkpoint_dir='checkpoints', log_interval=10)
    print(f"\nDone! Train Loss: {h['train_loss'][-1]:.4f}")
    if h['val_loss']: print(f"Val Loss: {h['val_loss'][-1]:.4f}")
    t.save_checkpoint('checkpoints/final_model.pt')
    plot_training_history(h, save_path='checkpoints/training_history.png')
    e = HiveMindEnvironment(num_bees=10, graph_generator=g)
    o = e.reset()
    d = Data(x=torch.tensor(o['node_features'], dtype=torch.float32),
             edge_index=torch.tensor(o['edge_index'], dtype=torch.long),
             edge_attr=torch.tensor(o['edge_attr'], dtype=torch.float32))
    t.model.eval()
    with torch.no_grad():
        ep, _ = t.model(d.x.to(t.device), d.edge_index.to(t.device), d.edge_attr.to(t.device))
    print(f"\nSample predictions: {ep[:10].cpu().numpy()}")
    plot_edge_predictions(ep.cpu().numpy(), save_path='checkpoints/edge_predictions.png')
    return t, h

if __name__ == '__main__':
    trainer, history = main()
