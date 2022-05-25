import unittest
import numpy as np
import torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1].resolve()))
from graph_sdc import graph_util


class TestCase(unittest.TestCase):
    
    def test_spacetime_knn_graph_trajectory(self):
        coord = th.tensor([[0, 0], [1, 0], [0, 2.5], [1, 2.5], [3, 0], [4, 0]], dtype=th.float)
        velocity = th.tensor([[1, 1], [-1, 1], [1, 1], [-1, 1], [1, 1], [-1, 1]], dtype=th.float)
        config = dict(seconds=1.0)
        expected = set([tuple(u) for u in [
            [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 2], [1, 3],
            [2, 0], [2, 1], [2, 3],
            [3, 0], [3, 1], [3, 2],
            [4, 0], [4, 1], [4, 5],
            [5, 0], [5, 1], [5, 4]]])
        
        edges = graph_util.spacetime_knn_graph(
            n_neighbors=3,
            coord=coord,
            velocity=velocity,
            metric="trajectory",
            config=config,
            loop=False)
        edges = set([tuple(u) for u in edges.T.tolist()])
        self.assertSetEqual(edges, expected)
    
    def test_spacetime_knn_graph_waypoints(self):
        coord = th.tensor(
            [[0, 0], [1, 0], [-1, 1], [0, 1], [3, 0], [4, 0]], dtype=th.float)
        velocity = th.tensor(
            [[1, 1], [-1, 1], [1, 1], [-1, 1], [1, 1], [-1, 1]],
            dtype=th.float)
        config = dict(seconds=1.0, sample_frequency=2, discount_factor=1.0)
        expected = set([tuple(u) for u in [
            [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 2], [1, 3],
            [2, 0], [2, 1], [2, 3],
            [3, 0], [3, 1], [3, 2],
            [4, 0], [4, 1], [4, 5],
            [5, 0], [5, 1], [5, 4]]])
        
        edges = graph_util.spacetime_knn_graph(
            n_neighbors=3,
            coord=coord,
            velocity=velocity,
            metric="waypoint",
            config=config,
            loop=False)
        edges = set([tuple(u) for u in edges.T.tolist()])
        self.assertSetEqual(edges, expected)
    
    def test_spacetime_knn_graph_insufficient_nodes(self):
        coord = th.rand((10, 2), dtype=th.float)
        velocity = th.rand_like(coord)
        n_neighbors = 10
        
        expected_with_loop = (2, 100)
        expected_without_loop = (2, 90)
        
        edge_with_loop = graph_util.spacetime_knn_graph(
            n_neighbors=n_neighbors,
            coord=coord,
            velocity=velocity,
            metric="trajectory",
            config=dict(seconds=1.0),
            loop=True)
        edge_without_loop = graph_util.spacetime_knn_graph(
            n_neighbors=n_neighbors,
            coord=coord,
            velocity=velocity,
            metric="trajectory",
            config=dict(seconds=1.0),
            loop=False)
        np.testing.assert_array_equal(expected_with_loop, edge_with_loop.shape)
        np.testing.assert_array_equal(
            expected_without_loop, edge_without_loop.shape)

if __name__ == "__main__":
    unittest.main()
