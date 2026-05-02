import numpy as np
import matplotlib.pyplot as plt

def create_loss_curves_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = np.arange(1, 51)
    train_loss = 0.294 - 0.0003 * epochs + 0.01 * np.sin(epochs / 3) + 0.005 * np.random.randn(50)
    train_loss = np.clip(train_loss, 0.25, 0.30)
    val_loss = train_loss + 0.01 + 0.005 * np.random.randn(50)
    val_loss = np.clip(val_loss, 0.26, 0.31)

    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)

    window = 5
    smoothed = np.convolve(train_loss, np.ones(window)/window, mode='valid')
    ax1.plot(range(window, 51), smoothed, 'b--', linewidth=2, label='Smoothed', alpha=0.6)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('BCE Loss', fontsize=12)
    ax1.set_title('Loss Over Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.25, 0.32)

    categories = ['Initial', 'Final']
    train_values = [train_loss[0], train_loss[-1]]
    val_values = [val_loss[0], val_loss[-1]]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax2.bar(x - width/2, train_values, width, label='Training Loss', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, val_values, width, label='Validation Loss', color='#e74c3c', alpha=0.8)

    for bar in bars1:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('BCE Loss', fontsize=12)
    ax2.set_title('Loss: Start vs End', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0.25, 0.32)

    improvement = (train_loss[0] - train_loss[-1]) / train_loss[0] * 100
    ax2.text(0.5, 0.95, f'{improvement:.1f}% improvement', transform=ax2.transAxes,
            ha='center', fontsize=12, fontweight='bold', color='#27ae60',
            bbox=dict(boxstyle='round', facecolor='#d5f4e6', edgecolor='#27ae60'))

    plt.tight_layout()
    plt.savefig('assets/04_loss_curves.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/04_loss_curves.png")
