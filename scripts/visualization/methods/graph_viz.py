import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_graph(graph, title="Flower Field Graph"):
    fig, ax = plt.subplots(figsize=(14, 10))

    pos = nx.spring_layout(graph, seed=42, k=2)

    node_colors = []
    node_sizes = []
    for i in range(graph.number_of_nodes()):
        if graph.nodes[i].get('is_source', False):
            node_colors.append('#FFD700')
            node_sizes.append(400)
        elif graph.nodes[i].get('is_sink', False):
            node_colors.append('#FF6B6B')
            node_sizes.append(400)
        else:
            nectar = graph.nodes[i].get('nectar_density', 0)
            node_colors.append(plt.cm.YlOrRd(nectar))
            node_sizes.append(150 + nectar * 200)

    edge_colors = []
    edge_widths = []
    for u, v in graph.edges():
        weight = graph.edges[u, v].get('base_weight', 1.0)
        edge_colors.append('#888888')
        edge_widths.append(0.5 + (1 / weight) * 0.5)

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=edge_widths, alpha=0.4, arrows=True, ax=ax)

    labels = {i: f'{i}' for i in range(graph.number_of_nodes()) if graph.nodes[i].get('is_source', False) or graph.nodes[i].get('is_sink', False)}
    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_weight='bold', ax=ax)

    source_patch = mpatches.Patch(color='#FFD700', label='Sources (Hive)')
    sink_patch = mpatches.Patch(color='#FF6B6B', label='Sinks (Flowers)')
    nectar_patch = mpatches.Patch(color='red', label='High Nectar', alpha=0.5)
    empty_patch = mpatches.Patch(color='wheat', label='Low Nectar', alpha=0.5)
    ax.legend(handles=[source_patch, sink_patch, nectar_patch, empty_patch], loc='upper left')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig, ax
