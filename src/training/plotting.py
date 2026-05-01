import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_training_history(history: Dict[str, List[float]], save_path: str = 'training_history.png'):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='blue', alpha=0.7)
    if history.get('val_loss'):
        axes[0].plot(epochs, history['val_loss'], label='Val Loss', color='orange', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss distribution (last 50 epochs)
    ax2 = axes[1]
    train_final = history['train_loss'][-50:] if len(history['train_loss']) > 50 else history['train_loss']
    ax2.hist(train_final, bins=20, alpha=0.5, label='Train', color='blue')
    if history.get('val_loss'):
        val_final = history['val_loss'][-50:] if len(history['val_loss']) > 50 else history['val_loss']
        ax2.hist(val_final, bins=20, alpha=0.5, label='Val', color='orange')
    ax2.set_xlabel('Loss')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Loss Distribution (Final Epochs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_edge_predictions(edge_probs: np.ndarray, edge_labels: np.ndarray = None, 
                         save_path: str = 'edge_predictions.png'):
    """Plot distribution of edge predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram of predictions
    axes[0].hist(edge_probs, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0].axvline(edge_probs.mean(), color='red', linestyle='--', 
                    label=f'Mean: {edge_probs.mean():.4f}')
    axes[0].set_xlabel('Edge Probability')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Edge Prediction Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # If labels available, show prediction vs truth
    if edge_labels is not None:
        axes[1].scatter(edge_labels, edge_probs, alpha=0.5, s=10)
        axes[1].set_xlabel('True Label')
        axes[1].set_ylabel('Predicted Probability')
        axes[1].set_title('Predictions vs Ground Truth')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].hist(edge_probs, bins=30, cumulative=True, density=True, 
                     alpha=0.7, color='purple', edgecolor='black')
        axes[1].set_xlabel('Edge Probability')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Distribution')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Edge predictions plot saved to {save_path}")
    plt.close()
