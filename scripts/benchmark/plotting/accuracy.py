import numpy as np
import matplotlib.pyplot as plt

def plot_gnn_accuracy_analysis(results, node_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dijkstra_costs = np.array(results['dijkstra']['costs'])
    gnn_costs = np.array(results['gnn']['costs'])

    if len(gnn_costs) > 0 and len(dijkstra_costs) > 0:
        valid_mask = (gnn_costs < float('inf')) & (dijkstra_costs[:len(gnn_costs)] < float('inf'))
        valid_mask = valid_mask[:len(gnn_costs)]
        dijkstra_valid = dijkstra_costs[:len(gnn_costs)][valid_mask]
        gnn_valid = gnn_costs[valid_mask]

        if len(gnn_valid) > 0:
            accuracy_ratios = gnn_valid / dijkstra_valid
            accuracy_pct = accuracy_ratios * 100

            axes[0].hist(accuracy_pct, bins=15, color='#9b59b6', edgecolor='black', alpha=0.7)
            axes[0].axvline(x=100, color='red', linestyle='--', linewidth=2, label='Dijkstra (100%)')
            axes[0].axvline(x=np.mean(accuracy_pct), color='green', linestyle='--', linewidth=2,
                            label=f'GNN Mean: {np.mean(accuracy_pct):.1f}%')
            axes[0].set_xlabel('GNN Cost / Dijkstra Cost (%)', fontsize=12)
            axes[0].set_ylabel('Count', fontsize=12)
            axes[0].set_title('GNN Accuracy Distribution', fontsize=14, fontweight='bold')
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

            x = np.arange(len(gnn_valid))
            width = 0.35
            axes[1].bar(x - width/2, dijkstra_valid, width, label='Dijkstra (Optimal)', color='#3498db', alpha=0.8)
            axes[1].bar(x + width/2, gnn_valid, width, label='GNN (Approximate)', color='#e74c3c', alpha=0.8)
            axes[1].set_xlabel('Trial', fontsize=12)
            axes[1].set_ylabel('Path Cost', fontsize=12)
            axes[1].set_title('GNN vs Dijkstra Path Costs', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3, axis='y')
        else:
            axes[0].text(0.5, 0.5, 'Insufficient GNN data\nfor accuracy analysis',
                        ha='center', va='center', fontsize=12)
            axes[1].text(0.5, 0.5, 'Insufficient GNN data\nfor comparison',
                        ha='center', va='center', fontsize=12)
    else:
        axes[0].text(0.5, 0.5, 'No valid path costs\nfound', ha='center', va='center', fontsize=12)
        axes[1].text(0.5, 0.5, 'No valid path costs\nfound', ha='center', va='center', fontsize=12)

    axes[0].set_xlim(50, 150)
    axes[0].axis('on')
    axes[1].axis('on')

    plt.tight_layout()
    plt.savefig('assets/10_gnn_accuracy.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: assets/10_gnn_accuracy.png")
