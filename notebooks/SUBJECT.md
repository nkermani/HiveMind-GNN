# HiveMind-GNN 🐝

*Neural Combinatorial Optimization for Autonomous Bee-Worker Routing*

## 📌 Overview

HiveMind-GNN is a research-oriented project exploring the intersection of Graph Theory and Deep Learning. Inspired by the multi-agent pathfinding challenges found in the "Lem-in" 42 curriculum, this project replaces static flow algorithms with Graph Neural Networks (GNNs) to solve routing problems in dynamic, uncertain environments.

The project simulates a colony of Bee Workers navigating a "Flower Field" (represented as a complex graph). The objective is to maximize nectar collection (throughput) while minimizing transit time and avoiding node congestion (collisions/bottlenecks).

## 🚀 The Challenge: Beyond Static Graphs

In traditional combinatorial optimization (like Edmonds-Karp or Dinic's algorithm), paths are calculated based on static edge capacities. However, real-world robotics—and Bee colonies—face uncertainty:

- **Variable Edge Weights**: Nectar levels change over time.
- **Node Congestion**: Dynamic bottlenecks occur when too many agents converge on a single node.
- **Scalability**: Optimal routing becomes NP-hard as the number of agents and nodes grows.

HiveMind-GNN handles this by using a Message Passing Neural Network (MPNN) to predict optimal edge weights, allowing agents to make decentralized, near-optimal routing decisions in real-time.

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| Logic & Environment | Custom Graph-Generator (C++ / Python) based on adjacency lists |
| Machine Learning | PyTorch Geometric (PyG) for Graph Convolutional Layers |
| Data Processing | Pandas & NumPy for simulation log analysis and feature engineering |

### Architecture

- **Encoder**: Embeds node features (nectar density, current occupancy).
- **Processor**: Multiple GNN layers to capture k-hop neighborhood information.
- **Decoder**: Predicts the probability of an edge being part of the "Global Optimal Path."

## 🧠 How it Works

1. **Graph Representation**: The "Flower Field" is modeled as a directed graph \( G = (V, E) \).
2. **Feature Attribution**: Each node \( v \) contains a feature vector \( x_v \) representing its current state.
3. **Neural Inference**: The GNN processes the graph to output a weight matrix \( W \) that accounts for predicted congestion.
4. **Flow Optimization**: A refined heuristic (inspired by the Ant Colony Optimization but guided by GNN weights) routes the bees through the field.

## 📈 Key "Research" Sell

> "This project demonstrates the ability to scale traditional algorithmic logic (Graph Flow) into the Neural Domain. By leveraging Graph Neural Networks, HiveMind-GNN moves away from rigid heuristics toward a model that 'learns' the topology of efficient routing, directly addressing the design bottlenecks of multi-agent fleet management."

## 📂 Project Structure

```
├── data/               # Generated Flower-Field datasets
├── notebooks/          # Exploratory Data Analysis & GNN Prototyping
├── src/
│   ├── env/            # Graph environment & Bee simulation logic
│   ├── model/          # PyTorch Geometric GNN architectures
│   └── train.py        # Training loops and loss function definitions
├── tests/              # Unit tests for graph integrity
└── README.md
```

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/nkermani/HiveMind-GNN.git

# Install dependencies
pip install torch torch-geometric pandas numpy
```
