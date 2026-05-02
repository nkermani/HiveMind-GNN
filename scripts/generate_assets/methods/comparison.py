import matplotlib.pyplot as plt

def create_comparison_table():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_title('Comparison with Alternatives', fontsize=14, fontweight='bold', pad=20)

    data = [
        ['Approach', 'Speed', 'Optimality', 'Adaptivity', 'Scalability'],
        ['Dijkstra', 'Slow (O(V\u00B2))', 'Optimal', 'None', 'Poor'],
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
