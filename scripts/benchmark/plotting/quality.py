import numpy as np
import matplotlib.pyplot as plt

def plot_quality_vs_speed(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    algorithms = ['dijkstra', 'astar', 'gnn']
    colors = {'dijkstra': '#3498db', 'astar': '#2ecc71', 'gnn': '#e74c3c'}
    labels = {'dijkstra': 'Dijkstra', 'astar': 'A*', 'gnn': 'GNN'}

    x_pos = 0
    positions = []
    times_list = []
    costs_list = []

    for algo in algorithms:
        times = np.array(results[algo]['times'])
        costs = np.array(results[algo]['costs'])
        positions.extend([x_pos, x_pos + 1, x_pos + 2])
        times_list.extend(times)
        costs_list.extend(costs)
        x_pos += 4

    bp1 = axes[0].boxplot(
        [results['dijkstra']['times'], results['astar']['times'], results['gnn']['times']],
        positions=[1, 2, 3], patch_artist=True
    )
    for patch, color in zip(bp1['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0].set_xticks([1, 2, 3])
    axes[0].set_xticklabels(['Dijkstra', 'A*', 'GNN'])
    axes[0].set_ylabel('Execution Time (ms)', fontsize=12)
    axes[0].set_title('Speed Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_yscale('log')

    bp2 = axes[1].boxplot(
        [results['dijkstra']['costs'], results['astar']['costs'], results['gnn']['costs']],
        positions=[1, 2, 3], patch_artist=True
    )
    for patch, color in zip(bp2['boxes'], ['#3498db', '#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(['Dijkstra', 'A*', 'GNN'])
    axes[1].set_ylabel('Path Cost', fontsize=12)
    axes[1].set_title('Quality Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('assets/09_quality_vs_speed.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Created: assets/09_quality_vs_speed.png")
