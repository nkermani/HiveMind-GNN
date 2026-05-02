import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_bee_navigation(graph, env, max_steps=20):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    pos = nx.spring_layout(graph, seed=42, k=2)

    node_colors = []
    for i in range(graph.number_of_nodes()):
        if graph.nodes[i].get('is_source', False):
            node_colors.append('#FFD700')
        elif graph.nodes[i].get('is_sink', False):
            node_colors.append('#FF6B6B')
        else:
            node_colors.append('#A8E6CF')

    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=200, alpha=0.7, ax=axes[0])
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.3, arrows=True, ax=axes[0])
    axes[0].set_title('Initial State', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    env.reset(graph)

    bee_positions = [bee.current_node for bee in env.bees]
    nx.draw_networkx_nodes(graph, pos, nodelist=bee_positions, node_color='blue', node_size=300, alpha=0.9, ax=axes[1])

    for idx, bee in enumerate(env.bees):
        x, y = pos[bee.current_node]
        axes[1].annotate(f'B{bee.bee_id}', (x, y), fontsize=8, ha='center', va='bottom', color='white', fontweight='bold')

    axes[1].set_title('Bees Placed at Sources', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    return fig
