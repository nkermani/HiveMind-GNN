import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_batch_of_graphs(generator):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx in range(6):
        gen = type(generator)(num_nodes=30, seed=idx * 10)
        g = gen.generate()

        pos = nx.spring_layout(g, seed=42, k=1.5)

        node_colors = []
        for i in range(g.number_of_nodes()):
            if g.nodes[i].get('is_source', False):
                node_colors.append('#FFD700')
            elif g.nodes[i].get('is_sink', False):
                node_colors.append('#FF6B6B')
            else:
                nectar = g.nodes[i].get('nectar_density', 0)
                node_colors.append(plt.cm.YlOrRd(nectar))

        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=80, alpha=0.8, ax=axes[idx])
        nx.draw_networkx_edges(g, pos, edge_color='gray', alpha=0.3, ax=axes[idx])
        axes[idx].set_title(f'Flower Field #{idx + 1}\n({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)', fontsize=10)
        axes[idx].axis('off')

    plt.tight_layout()
    return fig
