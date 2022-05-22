import sys
from pathlib import Path

import gym
sys.path.insert(0, str(Path(__file__).parents[1].resolve()))
from graph_sdc.env import HighwayEnvWrapper
import unittest

import numpy as np


class TestCase(unittest.TestCase):
    env_id = "highway-v0"
    config = {
        "observation": {
            "type": "Kinematics",
            "features": [
                "presence",
                "x",
                "y",
                "vx",
                "vy",
                "cos_h",
                "sin_h",
                "heading"
            ],
            "order": "sorted",
            "absolute": True,
            "normalize": True,
        }
    }

    def test_normalize_obs(self):
        env = gym.make(self.env_id)
        env = HighwayEnvWrapper(env, self.config)
        obs = np.array([[0, 100, 10, 30, 3, 0, 0, 0]], dtype=float)
        obs_dict = dict(zip(self.config["observation"]["features"], obs.T))
        obs_dict = env.normalize_obs(obs_dict)
        obs_normalized = [
            obs_dict[k] for k in self.config["observation"]["features"]]
        obs_normalized = np.stack(obs_normalized, axis=1)
        np.testing.assert_almost_equal(
            obs_normalized, [[0, 1.0, 0.1, 1.5, 0.15, 0, 0, 0]])

    def test_shift_obs(self):
        env = gym.make(self.env_id)
        env = HighwayEnvWrapper(env, self.config)
        obs = np.array([
            [0, 100, 10, 30, 3, 0, 0, 0],
            [0, 200, 10, 30, 3, 0, 0, 0]
        ], dtype=float)
        obs_dict = dict(zip(self.config["observation"]["features"], obs.T))
        obs_dict = env.shift_obs(obs_dict)
        obs_normalized = [
            obs_dict[k] for k in self.config["observation"]["features"]]
        obs_normalized = np.stack(obs_normalized, axis=1)
        np.testing.assert_almost_equal(
            obs_normalized,
            [
                [0, 0, 10, 30, 3, 0, 0, 0],
                [0, 100, 10, 30, 3, 0, 0, 0]
            ])


if __name__ == "__main__":
    unittest.main()
