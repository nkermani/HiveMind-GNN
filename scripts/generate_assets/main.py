import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.generate_assets.methods.pipeline import create_training_pipeline_figure
from scripts.generate_assets.methods.edge_probs import create_edge_probability_figure
from scripts.generate_assets.methods.flower_field import create_flower_field_figure
from scripts.generate_assets.methods.loss_curves import create_loss_curves_figure
from scripts.generate_assets.methods.features import create_feature_distributions_figure
from scripts.generate_assets.methods.architecture import create_architecture_diagram
from scripts.generate_assets.methods.comparison import create_comparison_table
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def run_generate_assets():
    print("Generating assets for HiveMind-GNN README...")
    print("-" * 40)

    create_training_pipeline_figure()
    create_edge_probability_figure()
    create_flower_field_figure()
    create_loss_curves_figure()
    create_feature_distributions_figure()
    create_architecture_diagram()
    create_comparison_table()

    print("-" * 40)
    print("All assets created successfully!")
    print("Location: assets/*.png")


if __name__ == '__main__':
    run_generate_assets()
