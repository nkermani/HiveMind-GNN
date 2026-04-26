from torch_geometric.loader import DataLoader as PyGDataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import numpy as np
import os
from typing import Optional, Dict, Tuple, List
import pickle

from src.env import FlowerFieldGenerator, HiveMindEnvironment
from src.model import HiveMindGNN, EdgePredictor


class GraphDataset(Dataset):
    def __init__(self, generator: FlowerFieldGenerator, num_samples: int = 1000):
        self.generator = generator
        self.num_samples = num_samples
        self.data_list = []
        self._generate_dataset()
    
    def _generate_dataset(self):
        for _ in range(self.num_samples):
            graph = self.generator.generate()
            label_dict = self.generator.compute_labels(graph)
            
            node_features = []
            for i in range(self.generator.num_nodes):
                feat = self.generator.get_feature_vector(i, graph)
                node_features.append(feat)
            
            edge_index_list = []
            edge_attr_list = []
            edge_labels_list = []
            
            for u, v in graph.edges():
                edge_index_list.append([u, v])
                w = self.generator.get_edge_weight(u, v, graph)
                edge_attr_list.append([w, graph.nodes[v].get('current_occupancy', 0)])
                edge_labels_list.append(label_dict.get((u, v), 0))
            
            data = Data(
                x=torch.tensor(np.array(node_features), dtype=torch.float32),
                edge_index=torch.tensor(np.array(edge_index_list).T, dtype=torch.long),
                edge_attr=torch.tensor(np.array(edge_attr_list), dtype=torch.float32),
                num_nodes=self.generator.num_nodes,
            )
            data.y = torch.tensor(np.array(edge_labels_list), dtype=torch.float32)
            self.data_list.append(data)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


class Trainer:
    def __init__(
        self,
        model: Optional[EdgePredictor] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model or EdgePredictor()
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, Data):
                batch = batch.to(self.device)
            else:
                continue
            
            self.optimizer.zero_grad()
            _, loss = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.y)
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, Data):
                    batch = batch.to(self.device)
                else:
                    continue
                
                _, loss = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.y)
                if loss is not None:
                    total_loss += loss.item()
                    num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10
    ) -> Dict[str, List[float]]:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
            
            if (epoch + 1) % log_interval == 0:
                msg = f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)
            
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        return {'train_loss': self.train_losses, 'val_loss': self.val_losses}
    
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Checkpoint loaded from {path}")


def main():
    print("=" * 60)
    print("HiveMind-GNN Training")
    print("=" * 60)
    
    generator = FlowerFieldGenerator(
        num_nodes=50,
        num_sources=5,
        num_sinks=5,
        density=0.3,
        seed=42
    )
    
    full_dataset = GraphDataset(generator, num_samples=500)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = PyGDataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = PyGDataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    model = EdgePredictor()
    trainer = Trainer(model, learning_rate=1e-3)
    
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    print("Starting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=100,
        checkpoint_dir='checkpoints',
        log_interval=10
    )
    
    print("\nTraining complete!")
    
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
    
    print(f"Final Train Loss: {final_train_loss:.4f}")
    if final_val_loss is not None:
        print(f"Final Val Loss: {final_val_loss:.4f}")
    
    trainer.save_checkpoint('checkpoints/final_model.pt')
    
    test_env = HiveMindEnvironment(num_bees=10, graph_generator=generator)
    obs = test_env.reset()
    
    node_features = torch.tensor(obs['node_features']).unsqueeze(0).to(trainer.device)
    edge_index = torch.tensor(obs['edge_index']).unsqueeze(0).to(trainer.device)
    edge_attr = torch.tensor(obs['edge_attr']).unsqueeze(0).to(trainer.device)
    
    trainer.model.eval()
    with torch.no_grad():
        edge_probs, _ = trainer.model(node_features, edge_index, edge_attr)
    
    print(f"\nSample edge predictions: {edge_probs[0][:10].cpu().numpy()}")
    
    return trainer, history


if __name__ == '__main__':
    trainer, history = main()