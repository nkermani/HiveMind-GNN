import numpy as np
import matplotlib.pyplot as plt

def plot_scalability_comparison(results, node_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for algo in ['dijkstra', 'astar', 'gnn']:
        times = np.array(results[algo]['times'])
        x_vals = np.linspace(0, len(node_sizes) - 1, len(times))
        axes[0].plot(x_vals, times, 'o-', label=algo.upper(), linewidth=2, markersize=6)

    axes[0].set_xticks(range(len(node_sizes)))
    axes[0].set_xticklabels(node_sizes)
    axes[0].set_xlabel('Number of Nodes', fontsize=12)
    axes[0].set_ylabel('Execution Time (ms)', fontsize=12)
    axes[0].set_title('Scalability: Execution Time vs Graph Size', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    dijkstra_costs = np.array(results['dijkstra']['costs'])
    axes[1].plot(range(len(dijkstra_costs)), dijkstra_costs, 'o-', label='Dijkstra (Optimal)', linewidth=2, markersize=6)

    if results['gnn']['costs']:
        gnn_costs = np.array(results['gnn']['costs'])
        axes[1].plot(range(len(gnn_costs)), gnn_costs, '^:', label='GNN (Approximate)', linewidth=2, markersize=6, alpha=0.8)

    axes[1].set_xlabel('Trial', fontsize=12)
    axes[1].set_ylabel('Path Cost', fontsize=12)
    axes[1].set_title('Solution Quality: Path Cost Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('assets/08_scalability_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: assets/08_scalability_comparison.png")
