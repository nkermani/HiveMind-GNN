import time
import numpy as np
import networkx as nx
import torch
from typing import List, Tuple, Optional


def dijkstra_shortest_path(graph, source, target):
    try:
        path = nx.dijkstra_path(graph, source, target, weight='base_weight')
        cost = nx.dijkstra_path_length(graph, source, target, weight='base_weight')
        return path, cost
    except nx.NetworkXNoPath:
        return None, float('inf')


def astar_shortest_path(graph, source, target):
    try:
        path = nx.astar_path(graph, source, target, heuristic=lambda u, v: 0, weight='base_weight')
        cost = nx.astar_path_length(graph, source, target, weight='base_weight')
        return path, cost
    except nx.NetworkXNoPath:
        return None, float('inf')


def gnn_predict_paths(model, graph, generator, sources, targets):
    node_features = []
    for i in range(generator.num_nodes):
        feat = generator.get_feature_vector(i, graph)
        node_features.append(feat)

    edge_index_list = []
    edge_attr_list = []
    for u, v in graph.edges():
        edge_index_list.append([u, v])
        w = generator.get_edge_weight(u, v, graph)
        edge_attr_list.append([w, 0])

    node_features = torch.tensor(np.array(node_features), dtype=torch.float32)
    edge_index = torch.tensor(np.array(edge_index_list).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_attr_list), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        edge_probs, _ = model(node_features, edge_index, edge_attr)

    edge_probs = edge_probs.cpu().numpy()
    gnn_paths = []
    for source, target in zip(sources, targets):
        path = []
        current = source
        visited = {source}
        while current != target and len(path) < generator.num_nodes:
            path.append(current)
            successors = list(graph.successors(current))
            valid = [s for s in successors if s not in visited]
            if not valid:
                break
            probs = []
            for s in valid:
                for idx, (u, v) in enumerate(graph.edges()):
                    if u == current and v == s:
                        probs.append((s, edge_probs[idx]))
                        break
            if not probs:
                break
            probs.sort(key=lambda x: x[1], reverse=True)
            next_node = probs[0][0]
            visited.add(next_node)
            current = next_node
        if path and path[-1] == target:
            gnn_paths.append((path, 0))
        else:
            gnn_paths.append((path, float('inf')))
    return gnn_paths


def compute_path_cost(graph, path):
    if not path or len(path) < 2:
        return float('inf')
    cost = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if graph.has_edge(u, v):
            cost += graph.edges[u, v].get('base_weight', 1.0)
        else:
            return float('inf')
    return cost