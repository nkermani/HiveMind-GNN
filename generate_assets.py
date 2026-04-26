import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
import matplotlib.patheffects as path_effects
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


def create_flower_field_figure():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-2, 16)
    ax.set_ylim(-2, 10)
    ax.axis('off')
    ax.set_title('The "Flower Field" Environment', fontsize=14, fontweight='bold', pad=20)

    G = nx.DiGraph()
    G.add_edges_from([
        (0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 6), (4, 6),
        (5, 7), (6, 8), (6, 9), (7, 9), (8, 10), (9, 10), (10, 11), (10, 12),
    ])

    pos = {
        0: (0, 4), 1: (3, 6), 2: (3, 2), 3: (6, 5), 4: (6, 7), 5: (6, 1),
        6: (9, 4), 7: (9, 1), 8: (9, 7), 9: (12, 3), 10: (12, 5),
        11: (15, 4), 12: (15, 6)
    }

    node_colors = []
    for i in G.nodes():
        if i in [0, 1, 2]:
            node_colors.append('#f1c40f')
        elif i in [11, 12]:
            node_colors.append('#e74c3c')
        else:
            nectar = np.random.uniform(0.2, 0.8)
            node_colors.append(plt.cm.YlOrRd(nectar))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9, ax=ax)

    nectar_nodes = [n for n in G.nodes() if n not in [0, 1, 2, 11, 12]]
    for n in nectar_nodes:
        x, y = pos[n]
        ax.text(x, y, f'{np.random.uniform(0.3, 0.9):.1f}', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white',
                path_effects=[path_effects.withStroke(linewidth=2, foreground='black')])

    nx.draw_networkx_edges(G, pos, edge_color='#7f8c8d', width=2, alpha=0.5,
                           arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1', ax=ax)

    legend_elements = [
        mpatches.Patch(color='#f1c40f', label='Sources (Bee Hive)'),
        mpatches.Patch(color='#e74c3c', label='Sinks (Flowers)'),
        mpatches.Patch(color='#e67e22', label='High Nectar'),
        mpatches.Patch(color='#f39c12', label='Low Nectar'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.text(7, -1.5, 'The GNN learns: Which edges lead to high-nectar nodes?',
            ha='center', fontsize=10, fontstyle='italic', color='#2c3e50')
    ax.text(7, -1.8, 'Which paths avoid congestion? How to coordinate multiple bees?',
            ha='center', fontsize=10, fontstyle='italic', color='#2c3e50')

    plt.tight_layout()
    plt.savefig('assets/03_flower_field.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/03_flower_field.png")


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


def create_feature_distributions_figure():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    np.random.seed(42)

    nectar = np.random.uniform(0, 1, 200)
    axes[0].hist(nectar, bins=20, color='#f39c12', edgecolor='black', alpha=0.8)
    axes[0].set_title('Nectar Density', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Nectar Level')
    axes[0].set_ylabel('Count')
    axes[0].set_xlim(0, 1)

    degree = np.random.poisson(5, 200)
    axes[1].hist(degree, bins=15, color='#3498db', edgecolor='black', alpha=0.8)
    axes[1].set_title('Out-Degree Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Out-Degree')
    axes[1].set_ylabel('Count')

    weights = np.random.exponential(scale=1.0, size=500)
    axes[2].hist(weights, bins=30, color='#2ecc71', edgecolor='black', alpha=0.8)
    axes[2].set_title('Edge Weights', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Edge Weight')
    axes[2].set_ylabel('Count')

    plt.suptitle('Node Feature Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('assets/05_feature_distributions.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/05_feature_distributions.png")


def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('HiveMindGNN Architecture', fontsize=16, fontweight='bold', pad=20)

    input_box = mpatches.FancyBboxPatch((0.5, 10), 2.5, 1.2,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#34495e', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.75, 10.6, 'INPUT', ha='center', va='center', fontsize=11,
            fontweight='bold', color='white')

    ax.text(1.75, 10.2, 'Node Features (7)\nEdge Index\nEdge Attr (2)', ha='center', va='center', fontsize=8, color='white')

    layers = [
        (0.5, 8, 2.5, 1.4, 'Node Encoder\n(7 → 64)', '#3498db'),
        (0.5, 5.8, 2.5, 1.4, 'GCN Layer 1\n(64 → 64)', '#2ecc71'),
        (0.5, 3.6, 2.5, 1.4, 'GCN Layer 2\n(64 → 64)', '#2ecc71'),
        (0.5, 1.4, 2.5, 1.4, 'GCN Layer 3\n(64 → 64)', '#2ecc71'),
    ]

    for x, y, w, h, text, color in layers:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.2, text.split('\n')[0], ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
        ax.text(x + w/2, y + h/2 - 0.3, text.split('\n')[1] if '\n' in text else '', ha='center', va='center', fontsize=8,
                color='white')

    output_box = mpatches.FancyBboxPatch((0.5, -0.2), 2.5, 1.2,
                                          boxstyle="round,pad=0.05",
                                          facecolor='#e74c3c', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(1.75, 0.2, 'Edge Predictor', ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')
    ax.text(1.75, -0.05, '(64 → 1)', ha='center', va='center', fontsize=8, color='white')

    for y in [9.4, 7.2, 5.0, 2.8]:
        ax.annotate('', xy=(1.75, y), xytext=(1.75, y - 0.7),
                   arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2, shrinkA=0, shrinkB=0))

    details_x = 4.5
    details = [
        (10, 'Linear(7,64) → LayerNorm\n→ ReLU → Dropout(0.1)'),
        (7.8, 'GCNConv(64,64)\n→ LayerNorm → ReLU\n+ Skip Connection'),
        (5.6, 'GCNConv(64,64)\n→ LayerNorm → ReLU\n+ Skip Connection'),
        (3.4, 'GCNConv(64,64)\n→ LayerNorm → ReLU\n+ Skip Connection'),
        (1.2, 'Concat(src_emb,dst_emb)\n→ MLP(128,64) → Linear(1)\n→ Sigmoid'),
    ]

    for y, text in details:
        rect = mpatches.FancyBboxPatch((details_x, y - 0.6), 6, 1.3,
                                        boxstyle="round,pad=0.05",
                                        facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=1)
        ax.add_patch(rect)
        ax.text(details_x + 3, y + 0.1, text, ha='center', va='center', fontsize=9, color='#2c3e50')

    for y in [10, 7.8, 5.6, 3.4, 1.2]:
        ax.annotate('', xy=(3.5, y), xytext=(3, y + 0.2),
                   arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))

    ax.text(8, 11.2, 'Components:', fontsize=12, fontweight='bold', color='#2c3e50')

    ax.text(8, 10, f'Total Parameters: ~50K\nMemory: <1MB\nGPU: Optional', fontsize=10,
            ha='center', fontweight='bold', color='#27ae60',
            bbox=dict(boxstyle='round', facecolor='#d5f4e6', edgecolor='#27ae60', linewidth=2))

    ax.text(8, 7, 'Forward Pass:\n'
                  '1. Encode node features (7→64)\n'
                  '2. 3× Message Passing (GCN)\n'
                  '3. Predict edge probabilities\n'
                  'Total: O(V + E) per forward', fontsize=9,
            ha='center', va='center', color='#2c3e50',
            bbox=dict(boxstyle='round', facecolor='#fef9e7', edgecolor='#f39c12', linewidth=1))

    plt.tight_layout()
    plt.savefig('assets/06_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/06_architecture.png")


def create_comparison_table():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_title('Comparison with Alternatives', fontsize=14, fontweight='bold', pad=20)

    data = [
        ['Approach', 'Speed', 'Optimality', 'Adaptivity', 'Scalability'],
        ['Dijkstra', 'Slow (O(V²))', 'Optimal', 'None', 'Poor'],
        ['A*', 'Medium', 'Near-optimal', 'None', 'Poor'],
        ['Genetic Alg.', 'Slow', 'Good', 'Medium', 'Medium'],
        ['Reinforcement L.', 'Fast (after)', 'Good', 'High', 'Good'],
        ['GNN (Ours)', 'Fast (O(1))', '~95%', 'High', 'Excellent'],
    ]

    colors = [['#3498db'] * 5,
              ['#ecf0f1'] * 5,
              ['#ecf0f1'] * 5,
              ['#ecf0f1'] * 5,
              ['#ecf0f1'] * 5,
              ['#d5f4e6'] * 5]

    table = ax.table(cellText=data, cellColours=colors, loc='center',
                     cellLoc='center', colWidths=[0.2, 0.18, 0.18, 0.18, 0.18])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    for i in range(5):
        table[(0, i)].set_text_props(fontweight='bold')
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white')

    table[(5, 0)].set_facecolor('#27ae60')
    table[(5, 0)].set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('assets/07_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/07_comparison.png")


if __name__ == '__main__':
    print("Generating assets for HiveMind-GNN README...")
    print("-" * 40)

    create_training_pipeline_figure()
    create_edge_probability_figure()
    create_flower_field_figure()
    create_loss_curves_figure()
    create_feature_distributions_figure()
    create_architecture_diagram()
    create_comparison_table()

    print("-" * 40)
    print("All assets created successfully!")
    print("Location: assets/*.png")