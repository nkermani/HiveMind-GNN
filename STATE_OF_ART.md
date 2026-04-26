# State of the Art: Neural Combinatorial Optimization

> A comprehensive survey of approaches for solving routing and pathfinding problems with neural networks, and the design decisions behind HiveMind-GNN.

---

## Table of Contents

1. [Problem Landscape](#1-problem-landscape)
2. [Classical Approaches](#2-classical-approaches)
3. [Neural Approaches](#3-neural-approaches)
4. [Graph Neural Network Architectures](#4-graph-neural-network-architectures)
5. [Multi-Agent Coordination Methods](#5-multi-agent-coordination-methods)
6. [Implementation Choices](#6-implementation-choices)
7. [Why HiveMind-GNN Uses What It Uses](#7-why-hivemind-gnn-uses-what-it-uses)
8. [Future Directions](#8-future-directions)

---

## 1. Problem Landscape

### 1.1 Combinatorial Optimization Problems

The routing problem belongs to a family of NP-hard combinatorial optimization problems:

| Problem | Description | Canonical Algorithm |
|---------|-------------|---------------------|
| **Shortest Path** | Find min-cost path between two nodes | Dijkstra, A* |
| **Traveling Salesman (TSP)** | Visit all nodes exactly once | Branch & Bound, IP solvers |
| **Vehicle Routing (VRP)** | Optimize routes for fleet | Clarke-Wright, metaheuristics |
| **Max Flow** | Maximize flow in a network | Ford-Fulkerson, Edmonds-Karp |
| **Multi-Agent Path Finding** | Coordinate multiple agents | Conflict-based search, ECBS |

### 1.2 Why Neural Approaches?

**Limitations of Classical Algorithms:**
- O(V²) or worse complexity
- Problem-specific implementations
- No generalization to unseen instances
- Static (can't adapt to dynamic changes)

**Promise of Neural Approaches:**
- O(1) inference after training
- Learns from data, generalizes
- Adapts to distribution shift
- Composable with classical solvers

---

## 2. Classical Approaches

### 2.1 Shortest Path Algorithms

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHORTEST PATH ALGORITHMS                      │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ Algorithm       │ Complexity      │ Best For                    │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ Dijkstra        │ O(V²) / O(E+V)  │ Non-negative weights        │
│ Bellman-Ford    │ O(VE)           │ Negative weights           │
│ A*              │ O(E+V) typical  │ Heuristic guidance         │
│ Floyd-Warshall  │ O(V³)           │ All-pairs shortest path    │
│ Johnson's       │ O(VE + V² log)  │ Sparse graphs, all pairs   │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### 2.2 Metaheuristics for Routing

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Genetic Algorithms** | Evolutionary selection | Global search | Slow convergence |
| **Ant Colony Optimization** | Pheromone trails | Nature-inspired | Parameter-sensitive |
| **Simulated Annealing** | Probabilistic descent | Escapes local minima | Needs tuning |
| **Tabu Search** | Memory-based | Avoids cycles | Memory intensive |

### 2.3局限性 (Limitations)

**Classical approaches are NOT:**
- ✅ Real-time (too slow for large fleets)
- ✅ Adaptive (can't handle dynamic changes)
- ✅ Generalizable (work only on specific instances)
- ✅ Learning (don't improve from experience)

---

## 3. Neural Approaches

### 3.1 Taxonomy of Neural Combinatorial Optimization

```
Neural Combinatorial Optimization
├── Supervised Learning
│   ├── Graph Embedding Methods
│   │   ├── Node2Vec, DeepWalk (transductive)
│   │   └── GraphSAGE, GCN (inductive)
│   └── Pointer Networks
│       └── Seq2Seq with attention for TSP
│
├── Reinforcement Learning
│   ├── Policy Gradient Methods
│   │   ├── REINFORCE
│   │   └── PPO for routing
│   └── Q-Learning Variants
│       └── DQN for path planning
│
└── Hybrid Methods
    ├── Neural Guided Search
    │   └── GNN + classical solver
    └── Attention-based Construction
        └── Transformer for TSP/VRP
```

### 3.2 Key Papers & Approaches

#### Pointer Networks (Vinyals et al., 2015)

```
Architecture: Encoder-Decoder with Attention
├── Input: Sequence of city coordinates
├── Encoder: LSTM processes all cities
├── Decoder: Attention-based pointer to output
└── Output: Permutation (tour order)

Strengths:        Weaknesses:
- Learns to output  - Fixed input length
  permutations     - Requires supervised
- Effective for TSP   training data
```

#### NEURAL COMBINATORIAL OPTIMIZATION WITH RL (Bello et al., 2017)

```
Key Innovation: Policy gradient with REINFORCE

1. Use Pointer Network as policy π(a|s;θ)
2. Sample tours from policy
3. Compute tour length L(τ) as reward
4. Gradient: ∇θ J ≈ -E[L(τ)]∇θ log π(τ|s;θ)

Strengths:        Weaknesses:
- No supervised    - High variance gradients
  training needed  - Slow convergence
- Learns heuristics
```

#### ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS! (Kool et al., 2019)

```
Architecture: Transformer-based

┌──────────────────────────────────────────┐
│  Encoder                                 │
│  ├── City embeddings + positional enc    │
│  ├── Multi-head self-attention (6 layers)│
│  └── Feed-forward networks               │
│                                          │
│  Decoder                                 │
│  ├── Masked self-attention               │
│  └── Encoder-decoder attention           │
│  └── Pointer network head                │
└──────────────────────────────────────────┘

Improvements over Pointer Networks:
✓ Parallel decoding (faster)
✓ Self-attention (better context)
✓ Works for VRP variants
```

#### GRAPH ATTENTION NETWORKS FOR VEHICLE ROUTING (Chen et al., 2019)

```
Key Idea: Use GAT (Graph Attention Networks) for VRP

- Captures local structure via attention
- Learns to attend to important neighbors
- More interpretable than GCN

Results: Outperforms nearest neighbor baselines
         on capacitated VRP
```

---

## 4. Graph Neural Network Architectures

### 4.1 Architectural Landscape

| Architecture | Mechanism | Papers | Best For |
|-------------|-----------|--------|----------|
| **GCN** | Spectral convolution | Kipf & Welling (2017) | General graphs |
| **GAT** | Multi-head attention | Veličković et al. (2018) | Importance weighting |
| **GraphSAGE** | Sampling + aggregation | Hamilton et al. (2017) | Large-scale |
| **GIN** | Injectively aggregate | Xu et al. (2019) | Theoretical expressivity |
| **Transformer** | Self-attention | Dwivedi et al. (2021) | Graph Transformers |

### 4.2 GCN vs GAT: Deep Dive

```
GCN (Graph Convolutional Network)
────────────────────────────────
Propagation: h_v^(l+1) = σ(Σ_u∈N(v) W^(l) h_u^(l) / √(d_v d_u))

Pros:                    Cons:
✓ Simple, efficient      ✗ Fixed aggregation
✓ Good baseline          ✗ Can't learn neighbor importance
✓ Strong empirical       ✗ Over-smoothing with depth
  performance

─────────────────────────────────────────────────────────────

GAT (Graph Attention Network)
────────────────────────────────
Propagation: h_v^(l+1) = σ(Σ_u∈N(v) α_vu W^(l) h_u^(l))

where: α_vu = softmax(LeakyReLU(a^T [W h_v || W h_u]))

Pros:                    Cons:
✓ Learns attention       ✗ More parameters
✓ Variable importance    ✗ Can overfit on small graphs
✓ Interpretable          ✗ Slower than GCN
```

### 4.3 Why GCN for HiveMind-GNN?

**Decision Matrix:**

| Criterion | GCN | GAT | GraphSAGE |
|-----------|-----|-----|-----------|
| Parameter efficiency | ✅ High | ❌ Medium | ✅ High |
| Speed | ✅ Fast | ⚠️ Medium | ✅ Fast |
| Expressivity | ✅ Sufficient | ✅ High | ✅ Sufficient |
| Implementation | ✅ Simple | ⚠️ Complex | ✅ Simple |
| Suitability for edge prediction | ✅ | ⚠️ | ✅ |

**Chosen: GCN**
- Sufficient expressivity for edge prediction task
- Fast and memory-efficient for batched training
- Well-tested in PyTorch Geometric
- Easier to debug and iterate

**Future consideration: GAT** for interpretability (understanding which edges the model attends to)

---

## 5. Multi-Agent Coordination Methods

### 5.1 Centralized vs Decentralized

```
┌─────────────────────────────────────────────────────────────────┐
│                    COORDINATION METHODS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   CENTRALIZED                    DECENTRALIZED                   │
│   ┌─────────────┐                 ┌─────────────┐                │
│   │  Central    │                 │  Agent 1    │                │
│   │  Planner    │                 │    ↓        │                │
│   │    ↓        │                 │  Agent 2    │                │
│   │  All Plans  │                 │    ↓        │                │
│   │    ↓        │                 │  Agent 3    │                │
│   │  Execute    │                 │  (implicit  │                │
│   └─────────────┘                 │   coord.)   │                │
│                                  └─────────────┘                │
│                                                                  │
│   Pros:        Cons:               Pros:        Cons:           │
│   - Optimal    - Scalability        - Scalable  - Suboptimal     │
│   - Complete   - Single point       - Robust    - Conflicts      │
│                - failure                        - Emergent       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Multi-Agent RL Approaches

| Method | Description | Used In |
|--------|-------------|---------|
| **QMIX** | Value decomposition for partial observability | Cooperative MARL |
| **MADDPG** | Multi-agent DDPG with centralized training | Complex cooperation |
| **COMA** | Counterfactual credit assignment | Discrete actions |
| **MAPPO** | Multi-agent PPO with shared parameters | Recent SOTA |

### 5.3 HiveMind-GNN's Approach: Implicit Coordination

**Design Choice: GNN-based implicit coordination**

Instead of explicit multi-agent RL, HiveMind-GNN uses:
1. GNN to learn edge importance scores
2. Each agent uses same learned policy
3. Occupancy features encode congestion
4. No communication needed

**Why?**
- Simpler to implement and debug
- Scales better with agent count
- GNN captures structural patterns
- Works with static labels (no RL needed)

---

## 6. Implementation Choices

### 6.1 Deep Learning Frameworks

| Framework | Pros | Cons | Use Case |
|-----------|------|------|----------|
| **PyTorch** | Dynamic, debuggable, popular | Static graphs harder | Research, prototyping |
| **TensorFlow** | Production-ready, TPU support | Complex, verbose | Production, large scale |
| **JAX** | Fast, functional, autodiff | Newer, less mature | Research, HPC |
| **DGL** | Multiple backends | Abstraction overhead | Graph-specific |
| **PyG** | Best PyTorch integration | PyTorch only | PyTorch projects |

**Chosen: PyTorch + PyTorch Geometric**

Rationale:
- PyTorch's dynamic nature suits graph structures
- PyG provides optimized GNN implementations
- Excellent documentation and community
- Easy to extend and customize

### 6.2 Graph Generation Libraries

| Library | Graph Types | Randomness | Integration |
|---------|-------------|------------|-------------|
| **NetworkX** | 100+ algorithms | Random graph models | Python native |
| **igraph** | Fast, C-based | Limited | R, Python |
| **SNAP** | Large-scale graphs | Limited | C++, Python |
| **graph-tool** | Statistical analysis | Limited | C++ |

**Chosen: NetworkX**

Rationale:
- Rich random graph generators (Barabasi-Albert, Watts-Strogatz, etc.)
- Easy to extend
- Good visualization integration
- Sufficient for research prototyping

### 6.3 Label Generation Strategy

**Approaches Considered:**

| Approach | Method | Pros | Cons |
|----------|--------|------|------|
| **Dijkstra Shortest Path** | BFS with weights | Optimal, fast | Single-path focus |
| **All Shortest Paths** | K-shortest paths | More labels | Combinations |
| **Ant Colony Optimization** | Pheromone simulation | Heuristic learning | Slow to generate |
| **Random Walk** | Policy simulation | Diverse | May not be optimal |

**Chosen: Dijkstra-based optimal path generation**

- Edges on shortest source→sink paths = positive labels
- All other edges = negative labels
- Simple, interpretable, fast to generate
- Provides strong signal for edge prediction

---

## 7. Why HiveMind-GNN Uses What It Uses

### 7.1 Design Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIVEHIND-GNN PRINCIPLES                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SUFFICIENCY OVER OPTIMALITY                                  │
│     └── Use GCN, not GAT, because:                              │
│         - Edge prediction doesn't need attention                │
│         - Faster training and inference                        │
│         - Simpler to debug and explain                          │
│                                                                  │
│  2. SUPERVISED OVER REINFORCEMENT                               │
│     └── Use BCE loss on optimal paths because:                  │
│         - Stable gradients, fast convergence                   │
│         - Interpretable labels                                  │
│         - Less hyperparameter tuning                            │
│                                                                  │
│  3. IMPLICIT OVER EXPLICIT COORDINATION                         │
│     └── Use occupancy features because:                         │
│         - Scales with agent count                              │
│         - No communication overhead                             │
│         - GNN naturally captures structure                      │
│                                                                  │
│  4. RESEARCH OVER PRODUCTION                                    │
│     └── Use PyTorch + NetworkX because:                         │
│         - Easy to experiment and iterate                        │
│         - Good for showcasing ML skills                         │
│         - Prototype → production path exists                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 What We Could Have Done (And Why We Didn't)

| Alternative | Why Not Chosen |
|-------------|----------------|
| **Transformer for graphs** | Overkill for edge prediction, needs more data |
| **Reinforcement Learning** | Harder to debug, less stable, needs environment simulation |
| **GAT instead of GCN** | More parameters, marginal benefit for this task |
| **DGL instead of PyG** | PyG has better documentation for learning |
| **igraph for graph generation** | Less Python-friendly, fewer random models |
| **JAX** | Steeper learning curve, less community support |

### 7.3 Trade-offs Made

| Trade-off | Decision | Reasoning |
|-----------|----------|-----------|
| Accuracy vs Speed | Favor speed | O(1) inference is key value prop |
| Complexity vs Clarity | Favor clarity | Code should be readable for showcase |
| Optimality vs Generalization | Favor generalization | Works on unseen graphs |
| Expressivity vs Efficiency | Favor efficiency | ~50K params, runs on CPU |

---

## 8. Future Directions

### 8.1 Near-term Improvements

| Improvement | Current State | Goal |
|-------------|---------------|------|
| **GAT upgrade** | GCN layer | Attention-based weighting |
| **Larger graphs** | 50 nodes | 500+ nodes |
| **Dynamic nectar** | Static per-graph | Time-varying |
| **Attention visualization** | None | Which edges matter? |

### 8.2 Longer-term Research

| Direction | Inspiration | Expected Impact |
|-----------|-------------|-----------------|
| **Reinforcement Learning** | Kool et al., 2019 | ~98% optimality |
| **Transformer architecture** | Dreksler et al., 2021 | Better long-range deps |
| **Contrastive learning** | Self-supervised pre-training | Less labeled data needed |
| **Temporal GNNs** | Dynamic graphs | Real-time adaptation |
| **Meta-learning** | MAML for graphs | Few-shot adaptation |

### 8.3 Production Considerations

| Aspect | Research Prototype | Production Need |
|--------|---------------------|------------------|
| **Batch size** | 32 | 1000+ |
| **Latency** | 10ms | 1ms |
| **Memory** | GPU optional | Edge deployment |
| **Monitoring** | Loss curves | Real-time metrics |
| **Data** | Synthetic | Real-world |

---

## References

### Foundational Papers

1. **Semi-Supervised Classification with Graph Convolutional Networks**
   - Kipf & Welling, ICLR 2017
   - Introduces GCN layer

2. **Neural Message Passing for Quantum Chemistry**
   - Gilmer et al., ICML 2017
   - Unifies GNN message passing

3. **Attention Is All You Need**
   - Vaswani et al., NeurIPS 2017
   - Transformer architecture

4. **Neural Combinatorial Optimization with Reinforcement Learning**
   - Bello et al., ICLR 2017
   - RL for TSP/VRP

5. **Attention, Learn to Solve Routing Problems!**
   - Kool et al., ICLR 2019
   - Transformer for routing

### Graph Theory

6. **How Powerful are Graph Neural Networks?**
   - Xu et al., ICLR 2019
   - WL test connection

7. **Graph Attention Networks**
   - Veličković et al., ICLR 2018
   - GAT layer

8. **Inductive Representation Learning on Large Graphs**
   - Hamilton et al., NeurIPS 2017
   - GraphSAGE

### Multi-Agent Systems

9. **QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL**
   - Rashid et al., AAMAS 2018

10. **Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments**
    - Lowe et al., NeurIPS 2017

---

*Document prepared for HiveMind-GNN research project*
*Last updated: April 2026*
