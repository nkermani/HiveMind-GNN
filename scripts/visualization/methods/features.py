import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def visualize_feature_distributions(graph, generator):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    nectar_levels = [graph.nodes[i].get('nectar_density', 0) for i in range(graph.number_of_nodes())]
    occupancy = [graph.nodes[i].get('current_occupancy', 0) for i in range(graph.number_of_nodes())]
    out_degree = [graph.out_degree(i) for i in range(graph.number_of_nodes())]
    in_degree = [graph.in_degree(i) for i in range(graph.number_of_nodes())]

    axes[0, 0].hist(nectar_levels, bins=20, color='#FFD700', edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('Nectar Density Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Nectar Level')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].hist(out_degree, bins=15, color='#4ECDC4', edgecolor='black', alpha=0.8)
    axes[0, 1].set_title('Out-Degree Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Out-Degree')
    axes[0, 1].set_ylabel('Count')

    axes[1, 0].hist(in_degree, bins=15, color='#FF6B6B', edgecolor='black', alpha=0.8)
    axes[1, 0].set_title('In-Degree Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('In-Degree')
    axes[1, 0].set_ylabel('Count')

    edge_weights = [generator.get_edge_weight(u, v, graph) for u, v in graph.edges()]
    axes[1, 1].hist(edge_weights, bins=30, color='#95E1D3', edgecolor='black', alpha=0.8)
    axes[1, 1].set_title('Edge Weight Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Edge Weight (lower = better)')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    return fig
