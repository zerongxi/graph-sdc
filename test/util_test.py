import unittest
import numpy as np
import torch as th

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1].resolve()))
from graph_sdc import util


class TestCase(unittest.TestCase):
    
    def test_check_line_intersect_2d(self):
        line1 = th.tensor([
            [[-1, 0], [1, 0]],
            [[-1, 0], [1, 0]],
            [[-1, 0], [1, 0]],
            [[-1, 0], [1, 0]],
        ], dtype=th.float)
        line1 = line1.permute((1, 2, 0))
        line2 = th.tensor([
            [[0, -1], [0, 1]],
            [[2, -1], [2, 1]],
            [[0, -1], [0, 0]],
            [[-1, 0], [1, 0]],
        ], dtype=th.float)
        line2 = line2.permute((1, 2, 0))
        expected = np.array([
            1.0,
            0.0,
            1.0,
            1.0,
        ], dtype=bool)
        
        intersect = util.check_line_intersect_2d(line1, line2)
        np.testing.assert_array_equal(intersect.numpy(), expected)
    
    def test_dist_point_to_line_2d(self):
        points = th.tensor([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [-1.0, 2.0]]).T
        lines = th.tensor([[-1.0, 1.0], [1.0, -1.0]], dtype=th.float).\
            unsqueeze(-1).repeat(1, 1, 5)
        sqrt2 = np.sqrt(2.)
        expected = th.tensor([
            [0.0, sqrt2 / 2.0, sqrt2, sqrt2, 1.0]],
            dtype=th.float).squeeze()
        
        dist = util.dist_point_to_line_2d(points, lines)
        np.testing.assert_array_almost_equal(dist, expected)
    
    def test_dist_between_lines_2d(self):
        line1 = th.tensor([
            [[-1, 0], [1, 0]],
            [[-1, 0], [1, 0]],
            [[-1, 0], [1, 0]],
            [[-1, 0], [1, 0]],
            [[-1, 1], [1, -1]],
        ], dtype=th.float)
        line1 = line1.permute((1, 2, 0))
        line2 = th.tensor([
            [[0, -1], [0, 1]],
            [[2, -1], [2, 1]],
            [[0, -1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [1, 2]],
        ], dtype=th.float)
        line2 = line2.permute((1, 2, 0))
        expected = np.array([
            0.0,
            1.0,
            0.0,
            0.0,
            np.sqrt(2.0) / 2.0,
        ], dtype=float)
        dist = util.dist_between_lines_2d(line1, line2)
        np.testing.assert_array_almost_equal(dist.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
