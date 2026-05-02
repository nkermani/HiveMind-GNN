import numpy as np
from typing import Dict, Optional
from .bee import Bee, BeeState
from .flower_field import FlowerFieldGenerator


class HiveMindEnvironment:
    def __init__(self, num_bees: int = 10, graph_generator: Optional[FlowerFieldGenerator] = None,
                 max_steps: int = 100):
        self.num_bees = num_bees
        self.graph_generator = graph_generator or FlowerFieldGenerator()
        self.max_steps = max_steps
        self.graph = None
        self.bees = []
        self.step_count = 0

    def reset(self, graph=None):
        self.graph = graph if graph is not None else self.graph_generator.generate()
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
        for bee, action in zip(self.bees, actions):
            if action < self.graph.number_of_nodes():
                bee.move_to(action, distance=np.random.random())
        return self._get_observation(), {}, self.step_count >= self.max_steps, {}
