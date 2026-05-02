import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
