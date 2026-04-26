import pytest
import numpy as np
import networkx as nx
from src.env import FlowerFieldGenerator, Bee, BeeState, HiveMindEnvironment


class TestFlowerFieldGenerator:
    def test_initialization(self):
        gen = FlowerFieldGenerator(num_nodes=50, seed=42)
        assert gen.num_nodes == 50
        assert gen.num_sources == 5
        assert gen.num_sinks == 5
        assert gen.density == 0.3

    def test_generate_returns_digraph(self):
        gen = FlowerFieldGenerator(num_nodes=30, seed=42)
        graph = gen.generate()
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 30

    def test_generate_has_sources_and_sinks(self):
        gen = FlowerFieldGenerator(num_nodes=30, num_sources=3, num_sinks=3, seed=42)
        graph = gen.generate()
        sources = [i for i in range(30) if graph.nodes[i].get('is_source', False)]
        sinks = [i for i in range(30) if graph.nodes[i].get('is_sink', False)]
        assert len(sources) == 3
        assert len(sinks) == 3
        assert len(set(sources) & set(sinks)) == 0

    def test_generate_has_nectar_levels(self):
        gen = FlowerFieldGenerator(num_nodes=30, seed=42)
        graph = gen.generate()
        for i in range(30):
            assert 'nectar_density' in graph.nodes[i]
            assert 0 <= graph.nodes[i]['nectar_density'] <= 1

    def test_generate_batch(self):
        gen = FlowerFieldGenerator(num_nodes=30, seed=42)
        graphs = gen.generate_batch(batch_size=5)
        assert len(graphs) == 5
        for g in graphs:
            assert isinstance(g, nx.DiGraph)

    def test_get_feature_vector(self):
        gen = FlowerFieldGenerator(num_nodes=30, seed=42)
        graph = gen.generate()
        feat = gen.get_feature_vector(0, graph)
        assert isinstance(feat, np.ndarray)
        assert feat.shape == (7,)
        assert feat.dtype == np.float32

    def test_get_edge_weight(self):
        gen = FlowerFieldGenerator(num_nodes=30, seed=42)
        graph = gen.generate()
        for u, v in list(graph.edges())[:5]:
            weight = gen.get_edge_weight(u, v, graph)
            assert isinstance(weight, float)
            assert weight > 0

    def test_generate_optimal_paths(self):
        gen = FlowerFieldGenerator(num_nodes=50, seed=42)
        graph = gen.generate()
        paths = gen.generate_optimal_paths(graph, num_paths=10)
        assert isinstance(paths, list)
        for path, cost in paths:
            assert isinstance(path, list)
            assert isinstance(cost, float)
            assert cost > 0

    def test_compute_labels(self):
        gen = FlowerFieldGenerator(num_nodes=40, seed=42)
        graph = gen.generate()
        labels = gen.compute_labels(graph)
        assert isinstance(labels, dict)
        for (u, v), label in labels.items():
            assert label in [0, 1]


class TestBee:
    def test_initialization(self):
        bee = Bee(bee_id=0, start_node=5)
        assert bee.bee_id == 0
        assert bee.current_node == 5
        assert bee.state == BeeState.IDLE
        assert bee.steps_taken == 0

    def test_move_to(self):
        bee = Bee(bee_id=0, start_node=0)
        bee.move_to(3, distance=2.5)
        assert bee.current_node == 3
        assert bee.steps_taken == 1
        assert bee.total_distance == 2.5
        assert bee.state == BeeState.NAVIGATING
        assert 3 in bee.path_history

    def test_collect_nectar(self):
        bee = Bee(bee_id=0, start_node=0)
        initial_nectar = bee.nectar_collected
        bee.collect_nectar(0.5)
        assert bee.nectar_collected == initial_nectar + 0.5
        assert bee.state == BeeState.COLLECTING

    def test_set_returning(self):
        bee = Bee(bee_id=0, start_node=0, target_node=5)
        bee.set_returning()
        assert bee.state == BeeState.RETURNING
        assert bee.target_node == 0

    def test_set_finished(self):
        bee = Bee(bee_id=0, start_node=0)
        bee.set_finished()
        assert bee.state == BeeState.FINISHED

    def test_is_stuck(self):
        bee = Bee(bee_id=0, start_node=0, max_steps=10)
        assert not bee.is_stuck()
        for _ in range(10):
            bee.move_to(bee.current_node, 0)
        assert bee.is_stuck()

    def test_reset(self):
        bee = Bee(bee_id=0, start_node=5, target_node=10)
        bee.move_to(6)
        bee.collect_nectar(0.5)
        bee.reset(start_node=3, target_node=7)
        assert bee.current_node == 3
        assert bee.target_node == 7
        assert bee.steps_taken == 0
        assert bee.nectar_collected == 0


class TestHiveMindEnvironment:
    def test_initialization(self):
        env = HiveMindEnvironment(num_bees=10)
        assert env.num_bees == 10
        assert env.max_steps == 200

    def test_reset(self):
        env = HiveMindEnvironment(num_bees=5)
        obs = env.reset()
        assert 'node_features' in obs
        assert 'edge_index' in obs
        assert 'edge_attr' in obs
        assert len(env.bees) == 5

    def test_reset_with_custom_graph(self):
        gen = FlowerFieldGenerator(num_nodes=30, seed=42)
        graph = gen.generate()
        env = HiveMindEnvironment(num_bees=3, graph_generator=gen)
        obs = env.reset(graph)
        assert obs['num_nodes'] == 30
        assert env.graph is graph

    def test_step(self):
        env = HiveMindEnvironment(num_bees=3, max_steps=10)
        env.reset()
        actions = np.array([0, 0, 0])
        obs, reward, done, info = env.step(actions)
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_multiple_steps(self):
        env = HiveMindEnvironment(num_bees=3, max_steps=5)
        env.reset()
        for _ in range(5):
            actions = np.random.randint(0, 5, size=3)
            obs, reward, done, info = env.step(actions)
            if done:
                break
        assert env.current_step >= 1

    def test_get_gnn_input(self):
        env = HiveMindEnvironment(num_bees=5)
        env.reset()
        node_feats, edge_idx, edge_attr = env.get_gnn_input()
        assert isinstance(node_feats, np.ndarray)
        assert isinstance(edge_idx, np.ndarray)
        assert isinstance(edge_attr, np.ndarray)

    def test_simulate_random_policy(self):
        env = HiveMindEnvironment(num_bees=5, max_steps=20)
        total_reward, bees = env.simulate_random_policy()
        assert isinstance(total_reward, float)
        assert len(bees) == 5

    def test_occupancy_tracking(self):
        env = HiveMindEnvironment(num_bees=3)
        env.reset()
        initial_occupancy = sum(
            env.graph.nodes[i].get('current_occupancy', 0) 
            for i in range(env.graph_generator.num_nodes)
        )
        assert initial_occupancy == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])