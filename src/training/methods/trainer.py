from torch_geometric.loader import DataLoader as PyGDataLoader
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import os
from typing import Optional, Dict, List
from src.model import EdgePredictor


class Trainer:
    def __init__(self, model: Optional[EdgePredictor] = None, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-5, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = (model or EdgePredictor()).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.train_losses = []
        self.val_losses = []

    def _process_batch(self, batch, train: bool = False) -> float:
        if not isinstance(batch, Data):
            return 0.0
        batch = batch.to(self.device)
        if train:
            self.optimizer.zero_grad()
        _, loss = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.y)
        if loss is not None and train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        return loss.item() if loss is not None else 0.0

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        losses, n = [], 0
        for batch in train_loader:
            loss = self._process_batch(batch, train=True)
            if loss > 0:
                losses.append(loss)
                n += 1
        return sum(losses) / max(1, n)

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        losses, n = [], 0
        with torch.no_grad():
            for batch in val_loader:
                loss = self._process_batch(batch, train=False)
                if loss > 0:
                    losses.append(loss)
                    n += 1
        return sum(losses) / max(1, n)

    def train(self, train_loader, val_loader=None, num_epochs=100, checkpoint_dir='checkpoints', log_interval=10):
        os.makedirs(checkpoint_dir, exist_ok=True)
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            val_loss = self.validate(val_loader) if val_loader else None
            if val_loss is not None:
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

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
