import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

def create_edge_probability_figure():
    fig, ax = plt.subplots(figsize=(10, 6))

    np.random.seed(42)
    n_edges = 150
    edge_probs = np.random.beta(2, 5, n_edges)
    edge_probs[:30] = np.random.beta(5, 2, 30)

    sorted_idx = np.argsort(edge_probs)[::-1]
    sorted_probs = edge_probs[sorted_idx]

    bars = ax.bar(range(n_edges), sorted_probs, color='#3498db', alpha=0.7, edgecolor='#2980b9')

    for i, bar in enumerate(bars[:20]):
        bar.set_color('#e74c3c')
        bar.set_alpha(0.9)

    ax.axhline(y=np.mean(edge_probs), color='#f39c12', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(edge_probs):.3f}')

    ax.set_xlabel('Edge Index (sorted by probability)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('GNN Edge Probability Predictions', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    high_patch = mpatches.Patch(color='#e74c3c', label='High prob (optimal path)')
    low_patch = mpatches.Patch(color='#3498db', label='Low prob (avoid)')
    ax.legend(handles=[high_patch, low_patch, Line2D([0], [0], color='#f39c12', linestyle='--', label='Mean')],
              loc='upper right')

    ax.set_ylim(0, 1.1)
    ax.set_xlim(-5, n_edges + 5)

    plt.tight_layout()
    plt.savefig('assets/02_edge_probabilities.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/02_edge_probabilities.png")
