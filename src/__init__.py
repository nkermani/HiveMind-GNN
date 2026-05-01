# src/__init__.py

from .env import FlowerFieldGenerator, Bee, BeeState, HiveMindEnvironment
from .model import HiveMindGNN, EdgePredictor, PathPredictor
from .training import Trainer, GraphDataset
from .training.plotting import plot_training_history, plot_edge_predictions

__version__ = "0.1.0"

__all__ = [
    'FlowerFieldGenerator', 'Bee', 'BeeState', 'HiveMindEnvironment',
    'HiveMindGNN', 'EdgePredictor', 'PathPredictor',
    'Trainer', 'GraphDataset',
    'plot_training_history', 'plot_edge_predictions',
]
