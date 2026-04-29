# tests/test_model.py

import pytest
import torch
import numpy as np
from src.model import HiveMindGNN, EdgePredictor, PathPredictor


class TestHiveMindGNN:
    def test_initialization(self):
        model = HiveMindGNN(
            node_input_dim=7,
            edge_input_dim=2,
            hidden_dim=64,
            num_layers=3
        )
        assert model.node_input_dim == 7
        assert model.hidden_dim == 64
        assert model.num_layers == 3

    def test_forward_pass(self):
        model = HiveMindGNN(hidden_dim=32, num_layers=2)

        num_nodes = 20
        num_edges = 50

        node_features = torch.randn(num_nodes, 7)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 2)

        embeddings, edge_logits = model(node_features, edge_index, edge_attr)

        assert embeddings.shape == (num_nodes, 32)
        assert edge_logits.shape == (num_edges,)

    def test_predict_edge_weights(self):
        model = HiveMindGNN(hidden_dim=32)

        node_features = torch.randn(20, 7)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.randn(50, 2)

        weights = model.predict_edge_weights(node_features, edge_index, edge_attr)

        assert weights.shape == (50,)
        assert torch.all(weights >= 0)
        assert torch.all(weights <= 1)

    def test_score_paths(self):
        model = HiveMindGNN(hidden_dim=32)

        node_embeddings = torch.randn(20, 32)
        paths = torch.tensor([
            [0, 1, 2, 3],
            [5, 6, 7, 8],
            [10, 11, 12, 13]
        ])

        scores = model.score_paths(node_embeddings, paths)

        assert scores.shape == (3,)
        assert torch.all(scores >= 0)
        assert torch.all(scores <= 1)


class TestEdgePredictor:
    def test_initialization(self):
        predictor = EdgePredictor()
        assert predictor.gnn is not None

    def test_forward_without_labels(self):
        predictor = EdgePredictor()

        node_features = torch.randn(20, 7)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.randn(50, 2)

        probs, loss = predictor(node_features, edge_index, edge_attr)

        assert probs.shape == (50,)
        assert loss is None

    def test_forward_with_labels(self):
        predictor = EdgePredictor()

        node_features = torch.randn(20, 7)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.randn(50, 2)
        labels = torch.randint(0, 2, (50,)).float()

        probs, loss = predictor(node_features, edge_index, edge_attr, labels)

        assert probs.shape == (50,)
        assert loss is not None
        assert isinstance(loss.item(), float)

    def test_predict_optimal_edges(self):
        predictor = EdgePredictor()

        node_features = torch.randn(20, 7)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.randn(50, 2)

        optimal_edges = predictor.predict_optimal_edges(node_features, edge_index, edge_attr, top_k=10)

        assert optimal_edges.shape == (50,)


class TestPathPredictor:
    def test_initialization(self):
        predictor = PathPredictor()
        assert predictor.gnn is not None

    def test_forward(self):
        predictor = PathPredictor()

        node_features = torch.randn(20, 7)
        edge_index = torch.randint(0, 20, (2, 50))
        edge_attr = torch.randn(50, 2)
        paths = torch.tensor([[0, 1, 2], [5, 6, 7]])

        scores = predictor(node_features, edge_index, edge_attr, paths)

        assert scores.shape == (2,)

    def test_select_best_paths(self):
        predictor = PathPredictor()

        node_features = torch.randn(30, 7)
        edge_index = torch.randint(0, 30, (2, 80))
        edge_attr = torch.randn(80, 2)
        candidate_paths = [[0, 1, 2, 3], [5, 6, 7, 8], [10, 11, 12, 13], [15, 16, 17, 18]]

        best_paths = predictor.select_best_paths(
            node_features, edge_index, edge_attr, candidate_paths, top_k=2
        )

        assert len(best_paths) <= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
