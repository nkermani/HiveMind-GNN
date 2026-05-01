import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional
from enum import Enum


class BeeState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    COLLECTING = "collecting"
    RETURNING = "returning"


class Bee:
    def __init__(self, bee_id: int, start_node: int):
        self.bee_id = bee_id
        self.current_node = start_node
        self.state = BeeState.IDLE
        self.steps_taken = 0
        self.total_distance = 0.0
        self.nectar_collected = 0.0
        self.path_history = [start_node]

    def move_to(self, node: int, distance: float):
        self.current_node = node
        self.steps_taken += 1
        self.total_distance += distance
        self.state = BeeState.NAVIGATING
        self.path_history.append(node)

    def collect_nectar(self, amount: float):
        self.nectar_collected += amount
        self.state = BeeState.COLLECTING


class FlowerFieldGenerator:
    def __init__(self, num_nodes: int = 50, num_sources: int = 5, num_sinks: int = 5,
                 density: float = 0.3, seed: Optional[int] = None):
        self.num_nodes = num_nodes
        self.num_sources = num_sources
        self.num_sinks = num_sinks
        self.density = density
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate(self) -> nx.DiGraph:
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_nodes))

        for i in range(self.num_nodes):
            G.nodes[i]['nectar_density'] = np.random.random()
            G.nodes[i]['is_source'] = False
            G.nodes[i]['is_sink'] = False
            G.nodes[i]['current_occupancy'] = 0

        sources = np.random.choice(self.num_nodes, self.num_sources, replace=False)
        sinks = np.random.choice([n for n in range(self.num_nodes) if n not in sources],
                                 self.num_sinks, replace=False)

        for s in sources:
            G.nodes[s]['is_source'] = True
        for t in sinks:
            G.nodes[t]['is_sink'] = True

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and np.random.random() < self.density:
                    G.add_edge(i, j, weight=np.random.random())

        return G

    def generate_batch(self, batch_size: int) -> List[nx.DiGraph]:
        return [self.generate() for _ in range(batch_size)]

    def get_feature_vector(self, node: int, graph: nx.DiGraph) -> np.ndarray:
        return np.array([
            graph.nodes[node].get('nectar_density', 0),
            1 if graph.nodes[node].get('is_source', False) else 0,
            1 if graph.nodes[node].get('is_sink', False) else 0,
            graph.nodes[node].get('current_occupancy', 0),
            len(list(graph.predecessors(node))),
            len(list(graph.successors(node))),
            graph.nodes[node].get('nectar_density', 0) * len(list(graph.successors(node)))
        ], dtype=np.float32)

    def get_edge_weight(self, u: int, v: int, graph: nx.DiGraph) -> float:
        return graph[u][v].get('weight', 1.0)

    def compute_labels(self, graph: nx.DiGraph) -> Dict[Tuple[int, int], int]:
        labels = {}
        optimal_paths = self.generate_optimal_paths(graph, num_paths=10)
        optimal_edges = set()
        for path, _ in optimal_paths:
            for i in range(len(path) - 1):
                optimal_edges.add((path[i], path[i+1]))

        for u, v in graph.edges():
            labels[(u, v)] = 1 if (u, v) in optimal_edges else 0
        return labels

    def generate_optimal_paths(self, graph: nx.DiGraph, num_paths: int = 10) -> List[Tuple[List[int], float]]:
        sources = [n for n in graph.nodes() if graph.nodes[n].get('is_source', False)]
        sinks = [n for n in graph.nodes() if graph.nodes[n].get('is_sink', False)]

        paths = []
        for s in sources[:num_paths]:
            for t in sinks[:num_paths]:
                try:
                    path = nx.shortest_path(graph, s, t, weight='weight')
                    cost = nx.shortest_path_length(graph, s, t, weight='weight')
                    paths.append((path, cost))
                except nx.NetworkXNoPath:
                    continue
        return sorted(paths, key=lambda x: x[1])[:num_paths]


class HiveMindEnvironment:
    def __init__(self, num_bees: int = 10, graph_generator: Optional[FlowerFieldGenerator] = None,
                 max_steps: int = 100):
        self.num_bees = num_bees
        self.graph_generator = graph_generator or FlowerFieldGenerator()
        self.max_steps = max_steps
        self.graph = None
        self.bees = []
        self.step_count = 0

    def reset(self):
        self.graph = self.graph_generator.generate()
        self.bees = [Bee(i, np.random.choice(list(self.graph.nodes()))) for i in range(self.num_bees)]
        self.step_count = 0
        return self._get_observation()

    def _get_observation(self):
        node_features = []
        for i in range(self.graph.number_of_nodes()):
            feat = self.graph_generator.get_feature_vector(i, self.graph)
            node_features.append(feat)

        edge_index = []
        edge_attr = []
        for u, v in self.graph.edges():
            edge_index.append([u, v])
            edge_attr.append([self.graph_generator.get_edge_weight(u, v, self.graph),
                             self.graph.nodes[v].get('current_occupancy', 0)])

        return {
            'node_features': np.array(node_features),
            'edge_index': np.array(edge_index).T if edge_index else np.empty((2, 0)),
            'edge_attr': np.array(edge_attr),
        }

    def step(self, actions):
        self.step_count += 1
        # Simplified step logic
        for bee, action in zip(self.bees, actions):
            if action < self.graph.number_of_nodes():
                bee.move_to(action, distance=np.random.random())
        return self._get_observation(), {}, self.step_count >= self.max_steps, {}
