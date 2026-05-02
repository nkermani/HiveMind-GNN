import numpy as np
import matplotlib.pyplot as plt

def create_feature_distributions_figure():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    np.random.seed(42)

    nectar = np.random.uniform(0, 1, 200)
    axes[0].hist(nectar, bins=20, color='#f39c12', edgecolor='black', alpha=0.8)
    axes[0].set_title('Nectar Density', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Nectar Level')
    axes[0].set_ylabel('Count')
    axes[0].set_xlim(0, 1)

    degree = np.random.poisson(5, 200)
    axes[1].hist(degree, bins=15, color='#3498db', edgecolor='black', alpha=0.8)
    axes[1].set_title('Out-Degree Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Out-Degree')
    axes[1].set_ylabel('Count')

    weights = np.random.exponential(scale=1.0, size=500)
    axes[2].hist(weights, bins=30, color='#2ecc71', edgecolor='black', alpha=0.8)
    axes[2].set_title('Edge Weights', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Edge Weight')
    axes[2].set_ylabel('Count')

    plt.suptitle('Node Feature Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('assets/05_feature_distributions.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Created: assets/05_feature_distributions.png")
