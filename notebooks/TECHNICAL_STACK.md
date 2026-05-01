# HiveMind-GNN Technical Stack

## Overview

This project showcases proficiency with modern deep learning and graph processing technologies.

---

## Technology Categories

### Deep Learning

| Technology | Usage | Level |
|------------|-------|-------|
| **PyTorch** | Neural network framework | Advanced |
| **PyTorch Geometric** | Graph neural networks | Advanced |
| **torch.optim** | Adam optimizer, LR scheduling | Intermediate |
| **torch.nn** | Custom layers, residual connections | Advanced |
| **autograd** | Automatic differentiation | Advanced |

### Graph Processing

| Technology | Usage | Level |
|------------|-------|-------|
| **NetworkX** | Graph generation (Barabasi-Albert) | Advanced |
| **PyTorch Geometric Data** | Graph data structures | Advanced |
| **GCNConv** | Message passing layers | Advanced |
| **PyG DataLoader** | Batched graph loading | Intermediate |

### Scientific Computing

| Technology | Usage | Level |
|------------|-------|-------|
| **NumPy** | Array operations, feature engineering | Advanced |
| **Pandas** | Data analysis | Intermediate |
| **Matplotlib** | Visualization | Advanced |
| **Seaborn** | Statistical plots | Intermediate |

### Development

| Technology | Usage | Level |
|------------|-------|-------|
| **pytest** | Unit testing | Intermediate |
| **git** | Version control | Advanced |
| **Jupyter** | Interactive notebooks | Intermediate |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                            │
│  ┌─────────┐  ┌─────────────┐  ┌──────────────────┐            │
│  │ Node    │  │ Edge Index  │  │ Edge Attributes  │            │
│  │ Features│  │ (src, dst)  │  │ (weight, occ)    │            │
│  │ (7-dim) │  │  (2, |E|)   │  │  (|E|, 2-dim)    │            │
│  └────┬────┘  └──────┬──────┘  └────────┬─────────┘            │
└───────┼──────────────┼─────────────────┼──────────────────────┘
        │              │                 │
        ▼              ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NODE ENCODER                              │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  Linear(7, 64) → LayerNorm → ReLU → Dropout(0.1)   │       │
│  └─────────────────────────────────────────────────────┘       │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MESSAGE PASSING LAYERS                       │
│                                                                 │
│   Layer 1: GCNConv(64, 64) → LayerNorm → ReLU                  │
│                     │                                           │
│                     ▼ (residual connection)                     │
│   Layer 2: GCNConv(64, 64) → LayerNorm → ReLU                  │
│                     │                                           │
│                     ▼ (residual connection)                     │
│   Layer 3: GCNConv(64, 64) → LayerNorm → ReLU                  │
│                                                                 │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EDGE PREDICTOR                             │
│                                                                 │
│   For each edge (u, v):                                         │
│     1. Get embeddings: emb_u, emb_v                             │
│     2. Concatenate: [emb_u; emb_v]                              │
│     3. MLP: Linear(128, 64) → ReLU → Linear(64, 1)             │
│     4. Sigmoid: probability of being on optimal path           │
│                                                                 │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                  │
│  Edge probabilities: [0.1, 0.8, 0.3, ..., p_n]                 │
│  Loss (if labels): BCE between predictions and ground truth     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Example

```python
# 1. Graph Generation (NetworkX)
graph = nx.barabasi_albert_graph(n=50, m=3, seed=42)
# Output: 50 nodes, ~150 edges

# 2. Feature Extraction (NumPy)
node_features = np.array([generator.get_feature_vector(i, graph) for i in range(50)])
# Output: shape (50, 7)

# 3. Convert to PyTorch Geometric format
data = Data(
    x=torch.tensor(node_features, dtype=torch.float32),
    edge_index=torch.tensor(edge_list, dtype=torch.long),
    edge_attr=torch.tensor(edge_features, dtype=torch.float32),
)
# Output: PyG Data object

# 4. Batch loading (PyG DataLoader)
loader = PyGDataLoader(dataset, batch_size=32, shuffle=True)
# Output: Batched graphs for GPU training

# 5. Forward pass (PyTorch)
model = HiveMindGNN().to('cuda')
embeddings, logits = model(data.x, data.edge_index, data.edge_attr)
# Output: (50, 64) embeddings, (150,) edge logits

# 6. Prediction
probs = torch.sigmoid(logits)
# Output: Edge probabilities [0, 1]
```

---

## Key Implementation Details

### GPU Utilization

```python
# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Move model and data to GPU
model = model.to(device)
batch = batch.to(device)

# GPU-accelerated forward pass
with torch.no_grad():
    probs = model(batch.x, batch.edge_index, batch.edge_attr)
```

### Efficient Batching

```python
# PyG automatically handles:
# - Variable graph sizes
# - Node-level vs graph-level features
# - Sparse adjacency matrices

loader = PyGDataLoader(dataset, batch_size=32, shuffle=True)
for batch in loader:
    # batch.x: (sum of nodes in batch, feature_dim)
    # batch.edge_index: (2, sum of edges in batch)
    # batch.batch: node-to-graph mapping
    pass
```

### Gradient Clipping

```python
# Prevent gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Testing Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| FlowerFieldGenerator | 12 tests | ✅ |
| Bee Agent | 7 tests | ✅ |
| HiveMindEnvironment | 8 tests | ✅ |
| HiveMindGNN | 4 tests | ✅ |
| EdgePredictor | 3 tests | ✅ |
| PathPredictor | 2 tests | ✅ |

Run with: `pytest tests/ -v`

---

## Reproducibility

```python
# Seed everything for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
```

---

*Technologies demonstrate readiness for ML engineering and research positions*