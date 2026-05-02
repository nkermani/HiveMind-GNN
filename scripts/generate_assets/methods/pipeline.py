import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def create_training_pipeline_figure():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('Training Pipeline', fontsize=14, fontweight='bold', pad=20)

    boxes = [
        (0.5, 2.5, 'Generate\n200 Graphs', '#3498db'),
        (4.5, 2.5, 'Extract\n7-dim Features', '#2ecc71'),
        (8.5, 2.5, 'Compute\nOptimal Paths', '#e74c3c'),
        (4.5, 0.8, 'Trained\nGNN Model', '#9b59b6'),
        (6.5, 1.8, 'Backprop\n+ Adam', '#f39c12'),
        (2.5, 1.8, 'Forward Pass\n+ BCE Loss', '#1abc9c'),
    ]

    for x, y, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), 2, 1.2, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + 1, y + 0.6, text, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')

    arrows = [
        (2.5, 3.1, 4.5, 3.1),
        (6.5, 3.1, 8.5, 3.1),
        (9.5, 2.5, 9.5, 2.0),
        (9.0, 2.0, 6.5, 2.0),
        (4.5, 2.0, 4.5, 1.8),
        (2.5, 2.0, 2.5, 1.8),
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    ax.text(10.5, 4.5, 'Loss: 0.294 → 0.273\n(7.4% improvement)', fontsize=10,
            ha='center', fontweight='bold', color='#27ae60',
            bbox=dict(boxstyle='round', facecolor='#d5f4e6', edgecolor='#27ae60'))

    plt.tight_layout()
    plt.savefig('assets/01_training_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/01_training_pipeline.png")
