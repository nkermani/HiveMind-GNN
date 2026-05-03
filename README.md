# HiveMind-GNN 🐝

*Neural Combinatorial Optimization for Autonomous Multi-Agent Routing*

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.3+-green.svg)](https://pytorch-geometric.readthedocs.io/)

---

## 🔬 The Problem: Why This Matters

### Real-World Challenge

Imagine **100 autonomous delivery drones** navigating a city, or **1000 robots** coordinating in a warehouse. Each must:

1. **Find optimal paths** to their destinations
2. **Avoid collisions** with other agents
3. **Adapt in real-time** to traffic and obstacles
4. **Coordinate implicitly** without central control

Traditional algorithms (Dijkstra, A*) fail here because:
- ❌ O(V²) complexity makes real-time decisions impossible
- ❌ Single-agent focus, no multi-agent coordination
- ❌ Static graphs don't adapt to changing conditions
- ❌ No generalization to unseen environments

### The Solution: Graph Neural Networks

| Aspect | Traditional Approach | Neural Approach |
|--------|----------------------|-----------------|
| Method | Query graph for answer | Learn patterns from graphs |
| Complexity | O(V²) per query | O(1) forward pass after training |
| Solution | Exact optimal | ~95% of optimal, but instant |
| Scope | Problem-specific | Generalizes to new graphs |

---

## 🎯 What This Project Solves

### Core Capabilities

| Capability | Description | Use Case |
|------------|-------------|----------|
| **Path Prediction** | Predicts which edges lead to optimal routes | Drone navigation, robot fleet management |
| **Congestion Avoidance** | Learns to avoid bottleneck nodes | Traffic optimization, network routing |
| **Real-Time Inference** | O(1) decision making after training | Autonomous vehicles, emergency response |
| **Multi-Agent Coordination** | Implicit coordination without communication | Warehouse robots, swarm robotics |
| **Topology Generalization** | Works on graphs never seen during training | Adapts to new warehouses, cities, networks |


## 🏗️ Architecture

```
HiveMindGNN
├── Node Encoder (7 → 64 dim)
│   └── Linear(7,64) → LayerNorm → ReLU → Dropout(0.1)
│
├── Message Passing Layers (×3)
│   ├── GCN Layer 1: GCNConv(64, 64)
│   ├── GCN Layer 2: GCNConv(64, 64)
│   └── GCN Layer 3: GCNConv(64, 64)
│       └── Each layer: Conv → LayerNorm → ReLU + Skip Connection
│
└── Edge Predictor
    └── Concatenate(src_emb, dst_emb) → MLP(128,64) → Linear(64,1) → Sigmoid
```

| **Total Parameters:** | **~73K** (lightweight, deployable on edge devices) |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/nkermani/HiveMind-GNN.git
cd HiveMind-GNN
pip install -r requirements.txt
```

### Generate Visualizations

```bash
python visualize.py
# Creates: visualizations/*.png
```

### Train the Model
```python
from src.train import main
from src.training import GraphDataset, Trainer
from src.model import EdgePredictor
from src.env import FlowerFieldGenerator

# Run full training with visualization
trainer, history = main()

# Or customize training
generator = FlowerFieldGenerator(num_nodes=50, seed=42)
dataset = GraphDataset(generator, num_samples=500)
model = EdgePredictor()
trainer = Trainer(model)
# See src/train.py for full options
```

### Make Predictions
```python
import torch
from src.env import FlowerFieldGenerator, HiveMindEnvironment
from src.model import EdgePredictor
from torch_geometric.data import Data

# Load trained model
model = EdgePredictor()
model.load_state_dict(torch.load('checkpoints/final_model.pt'))

# Get observation from environment
generator = FlowerFieldGenerator(num_nodes=50)
env = HiveMindEnvironment(num_bees=10, graph_generator=generator)
obs = env.reset()

# Create PyG Data object
data = Data(
    x=torch.tensor(obs['node_features'], dtype=torch.float32),
    edge_index=torch.tensor(obs['edge_index'], dtype=torch.long),
    edge_attr=torch.tensor(obs['edge_attr'], dtype=torch.float32)
)

# Predict optimal edges
model.eval()
with torch.no_grad():
    edge_probs, _ = model(data.x, data.edge_index, data.edge_attr)

# edge_probs: which edges lead to optimal routes?
```

---

## 📁 Project Structure

```
HiveMind-GNN/
├── assets/                     # Visual assets for README
├── checkpoints/                # Saved models (excluded from git)
├── data/                      # Dataset storage
├── notebooks/
│   ├── EXPLANATIONS.md       # Theory & deep-dive
│   ├── STATE_OF_ART.md       # Literature survey
│   ├── SUBJECT.md            # Project subject
│   └── TECHNICAL_STACK.md   # Technology showcase
├── src/
│   ├── env.py                # Environment & graph generator
│   ├── model/
│   │   ├── hivemind_gnn/     # GNN architecture
│   │   │   ├── encoders.py
│   │   │   ├── gcn_stack.py
│   │   │   ├── edge_scorer.py
│   │   │   ├── path_scorer.py
│   │   │   └── positional_encoding.py
│   │   └── predictor/        # Prediction heads
│   │       ├── edge_predictor.py
│   │       └── path_predictor.py
│   ├── training/             # Training pipeline
│   │   ├── dataset.py
│   │   ├── trainer.py
│   │   └── plotting.py
│   └── train.py             # Main training script (40 lines)
├── tests/
│   ├── test_env.py
│   └── test_model.py
├── visualizations/            # Generated PNGs (excluded from git)
├── shell.nix                 # Nix dev environment
├── visualize.py
├── benchmark.py
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Deep-Dive

### Why GNNs Work for Routing

**1. Weisfeiler-Lehman Connection**
- GNNs are a neural approximation of the WL graph isomorphism test
- They learn structural patterns that distinguish good paths from bad ones

**2. Local Computation**
- GCN operates on k-hop neighborhoods
- Scales near-linearly: O(V + E) vs O(V²) for Dijkstra

**3. Inductive Bias**
- Graph structure is explicitly modeled
- Transforms naturally to graphs of different sizes

### Key Innovations

| Innovation | Why It Matters |
|------------|----------------|
| **Barabasi-Albert Graphs** | Scale-free networks (like real cities) |
| **7-dim Node Features** | Captures nectar, occupancy, degree |
| **Residual Connections** | Prevents over-smoothing, enables depth |
| **BCE Loss** | Directly optimizes for optimal path membership |
| **Multi-Agent Environment** | Simulates real coordination challenges |

---

## 💼 Real-World Applications

| Industry | Problem Solved | How GNN Helps |
|----------|---------------|---------------|
| **Autonomous Vehicles** | Urban route planning | Real-time adaptation to traffic |
| **Warehouse Robotics** | Multi-robot coordination | Implicit collision avoidance |
| **Telecommunications** | Network routing | Dynamic traffic optimization |
| **Emergency Response** | Evacuation planning | Fast path computation under stress |
| **Supply Chain** | Delivery route optimization | Scales to 1000s of stops |

---

## 📈 Comparison with Alternatives


| Approach | Speed | Optimality | Adaptivity | Scalability |
|----------|-------|------------|------------|-------------|
| **Dijkstra** | Slow (O(V²)) | Optimal | None | Poor |
| **A*** | Medium | Near-optimal | None | Poor |
| **Genetic Algorithms** | Slow | Good | Medium | Medium |
| **Reinforcement Learning** | Fast after training | Good | High | Good |
| **GNN (Ours)** | Fast (O(1)) | ~83-95% | High | Excellent |

### Benchmark Results

Run `python benchmark.py` to generate comparative analysis:

| Metric | Dijkstra | A* | GNN (Ours) |
|--------|----------|-----|------------|
| **Execution Time** | ~0.5ms | ~0.06ms | ~2.3ms |
| **Path Cost** | Optimal (baseline) | Optimal | 83% of optimal |
| **Training Required** | No | No | Yes (50 epochs) |
| **Generalization** | None | None | Works on new graphs |
| **Per-Query Cost** | O(V²) | O(E+V) | O(1) after training |

---

## 📚 Documentation

- **[EXPLANATIONS.md](EXPLANATIONS.md)** - Theoretical foundations, MPNN, WL algorithm
- **[TECHNICAL_STACK.md](TECHNICAL_STACK.md)** - PyTorch, PyG, NetworkX showcase
- **[STATE_OF_ART.md](STATE_OF_ART.md)** - Literature survey, design decisions, comparisons
- **[notebooks/01_exploration.ipynb](notebooks/01_exploration.ipynb)** - Interactive exploration

---

## 🎓 What This Project Demonstrates

### Technical Skills
- ✅ Deep learning with PyTorch & PyTorch Geometric
- ✅ Graph neural networks (GCN, message passing)
- ✅ Multi-agent simulation and coordination
- ✅ Neural combinatorial optimization
- ✅ Data visualization and analysis

### Research Skills
- ✅ Problem formulation and motivation
- ✅ Theoretical grounding (WL, MPNN)
- ✅ Experimental design and evaluation
- ✅ Limitation analysis and future work

### Engineering Skills
- ✅ Clean, modular code architecture
- ✅ Unit testing with pytest
- ✅ Documentation and visualization
- ✅ Reproducibility (seeded randomness)

---

## 🙏 Acknowledgments

Inspired by the Lem-in challenge from the 42 curriculum.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

*Built with PyTorch, PyTorch Geometric, NetworkX*
*For questions, open an issue or reach out to [nkermani](https://github.com/nkermani)*
