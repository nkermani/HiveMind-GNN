import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from src.env import FlowerFieldGenerator


class GraphDataset(Dataset):
    def __init__(self, generator: FlowerFieldGenerator, num_samples: int = 1000):
        self.generator = generator
        self.num_samples = num_samples
        self.data_list = []
        self._generate_dataset()

    def _generate_dataset(self):
        for _ in range(self.num_samples):
            graph = self.generator.generate()
            label_dict = self.generator.compute_labels(graph)
            nodes = self.generator.num_nodes

            node_features = np.array([self.generator.get_feature_vector(i, graph) for i in range(nodes)], dtype=np.float32)
            edges = list(graph.edges())
            edge_index_list = [[u, v] for u, v in edges]
            edge_attr_list = [[float(self.generator.get_edge_weight(u, v, graph)),
                               float(graph.nodes[v].get('current_occupancy', 0))] for u, v in edges]
            edge_labels = [int(label_dict.get((u, v), 0)) for u, v in edges]

            data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=torch.tensor(edge_index_list, dtype=torch.long).T,
                edge_attr=torch.tensor(edge_attr_list, dtype=torch.float32),
                num_nodes=nodes,
            )
            data.y = torch.tensor(edge_labels, dtype=torch.float32)
            self.data_list.append(data)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]
