import numpy as np
import matplotlib.pyplot as plt

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
