import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects

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
