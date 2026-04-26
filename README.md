# HiveMind-GNN

*Neural Combinatorial Optimization for Autonomous Bee-Worker Routing*

## Overview

HiveMind-GNN is a research-oriented project that uses Graph Neural Networks (GNNs) to solve multi-agent routing problems in dynamic, uncertain environments. Inspired by bee colonies navigating flower fields, it replaces static flow algorithms with neural networks to maximize nectar collection while minimizing transit time and avoiding congestion.

## Installation

```bash
git clone https://github.com/nkermani/HiveMind-GNN.git
cd HiveMind-GNN
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch >= 2.0
- PyTorch Geometric >= 2.3
- NetworkX >= 3.0
- NumPy, Pandas, Matplotlib

## Quick Start

### 1. Generate a Flower Field Graph

```python
from src.env import FlowerFieldGenerator

generator = FlowerFieldGenerator(
    num_nodes=50,      # Number of nodes in the graph
    num_sources=5,     # Starting nodes for bees
    num_sinks=5,       # Target nodes for bees
    density=0.3,       # Edge density
    seed=42
)

graph = generator.generate()
print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

### 2. Simulate the Environment

```python
from src.env import HiveMindEnvironment

env = HiveMindEnvironment(num_bees=10, graph_generator=generator)
obs = env.reset(graph)  # Reset with your graph

# Take actions (one action per bee - selecting next node from successors)
actions = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]  # Example actions
obs, reward, done, info = env.step(actions)

print(f"Observation keys: {obs.keys()}")
print(f"Reward: {reward:.2f}")
print(f"Info: {info}")
```

### 3. Train the GNN Model

```python
from src.train import Trainer, GraphDataset
from src.model import EdgePredictor
from torch_geometric.loader import DataLoader as PyGDataLoader

generator = FlowerFieldGenerator(num_nodes=50, seed=42)
dataset = GraphDataset(generator, num_samples=500)

train_loader = PyGDataLoader(dataset, batch_size=32, shuffle=True)

model = EdgePredictor()
trainer = Trainer(model, learning_rate=1e-3)

history = trainer.train(train_loader, num_epochs=100, log_interval=10)
```

### 4. Make Predictions

```python
import torch

# Get observation from environment
obs = env.reset()
node_features = torch.tensor(obs['node_features'])
edge_index = torch.tensor(obs['edge_index'])
edge_attr = torch.tensor(obs['edge_attr'])

# Predict optimal edges
model.eval()
with torch.no_grad():
    edge_probs, _ = model(node_features, edge_index, edge_attr)

print(f"Top 10 edge probabilities: {edge_probs[:10].numpy()}")
```

## Core Components

### `FlowerFieldGenerator`

Generates directed graphs simulating flower fields with:
- **Nectar density**: Node features representing nectar availability
- **Source/Sink nodes**: Start and end points for bee routing
- **Dynamic edge weights**: Based on nectar and congestion

```python
# Methods
graph = generator.generate()                      # Generate single graph
graphs = generator.generate_batch(batch_size=10)  # Generate multiple graphs
paths = generator.generate_optimal_paths(graph)   # Get reference paths
labels = generator.compute_labels(graph)          # Generate training labels
```

### `HiveMindEnvironment`

Multi-agent simulation environment with:
- **State tracking**: Occupancy, nectar collection, step counts
- **Reward shaping**: Nectar collected, efficiency, congestion penalty
- **GNN-ready observations**: Node features, edge indices, edge attributes

```python
# Methods
obs = env.reset()                    # Initialize environment
obs, reward, done, info = env.step(actions)  # Take action
total_reward, bees = env.simulate_random_policy()  # Baseline evaluation
```

### `HiveMindGNN`

Graph Neural Network with:
- **Node encoder**: 7-dimensional input features
- **GCN layers**: 3 layers with skip connections
- **Edge predictor**: Scores edges for optimal routing

```python
model = HiveMindGNN(
    node_input_dim=7,
    edge_input_dim=2,
    hidden_dim=64,
    num_layers=3,
    dropout=0.1
)
```

### `EdgePredictor`

Wrapper model for edge classification:
- Predicts probability of edges being part of optimal paths
- Computes BCE loss when labels provided

## Training Pipeline

```python
from src.train import Trainer, GraphDataset
from src.model import EdgePredictor
from torch_geometric.loader import DataLoader as PyGDataLoader

gen = FlowerFieldGenerator(num_nodes=50, seed=42)
dataset = GraphDataset(gen, num_samples=500)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = PyGDataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = PyGDataLoader(val_ds, batch_size=32)

model = EdgePredictor()
trainer = Trainer(model)

history = trainer.train(
    train_loader,
    val_loader,
    num_epochs=100,
    checkpoint_dir='checkpoints',
    log_interval=10
)

trainer.save_checkpoint('checkpoints/final_model.pt')
```

## Jupyter Notebooks

Explore the project interactively:

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

Features:
- Graph visualization
- Feature distribution analysis
- Environment simulation
- GNN inference testing

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
HiveMind-GNN/
├── data/                   # Dataset storage
├── notebooks/
│   └── 01_exploration.ipynb
├── src/
│   ├── env/
│   │   ├── bee.py          # Bee agent class
│   │   ├── environment.py  # Simulation environment
│   │   └── graph_generator.py
│   ├── model/
│   │   ├── hivemind_gnn.py   # GNN architecture
│   │   └── edge_predictor.py # Edge prediction wrapper
│   └── train.py           # Training loop
├── tests/
│   ├── test_env.py
│   └── test_model.py
├── requirements.txt
├── setup.py
└── README.md
```

## Citation

If you use HiveMind-GNN in your research:

> Kermani, N. (2024). HiveMind-GNN: Neural Combinatorial Optimization for Autonomous Bee-Worker Routing. Neural Combinatorial Optimization Research Track.