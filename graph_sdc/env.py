from copy import deepcopy
from datetime import datetime
import logging
from typing import Dict, Optional, Tuple, Union
import gym
from gym import spaces
import numpy as np
import torch as th
import highway_env

from torch_geometric.data import Data as Graph

from graph_sdc.data import Vehicle2D, obs2vehicle2d, get_node_attr, get_edge_attr
from graph_sdc.graph_util import build_graph
from graph_sdc.util import linear_map


class HighwayEnv(gym.Env):
    def __init__(self, env_id: str, config: Dict) -> None:
        logging.info("Use customized highway environment")
        config = deepcopy(config)
        self.config = deepcopy(config)
        self.normalize = config["observation"].get("normalize", True)
        self.features = config["observation"]["features"]
        self.features_range = config["observation"].get("features_range", {
            "x": [0, 100],
            "y": [0, 100],
            "vx": [0, 20],
            "vy": [0, 20],
        })
        self.clip = config["observation"].get("clip", False)
        self.absolute = config["observation"].get("absolute", False)

        self.env = gym.make(env_id)
        config["observation"]["normalize"] = False
        self.env.configure(config)
        self.env.reset()
        
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def normalize_obs(self, obs: Dict) -> Dict:
        for feat in self.features_range:
            if feat in obs.keys():
                obs[feat] = linear_map(
                    obs[feat], self.features_range[feat], [0.0, 1.0],)
        return obs

    def shift_obs(self, obs: Dict) -> Dict:
        if self.absolute:
            obs["x"] = obs["x"] - obs["x"][0]
        obs["x"][0] = 0.0
        return obs
    
    def observe(self, obs: np.ndarray) -> np.ndarray:
        obs_dict = dict(zip(self.features, obs.T))
        obs_dict = self.shift_obs(obs_dict)
        if self.normalize:
            obs_dict = self.normalize_obs(obs_dict)
        obs = [obs_dict[k] for k in self.features]
        obs = np.stack(obs, axis=1)
        return obs
    
    def reset(self):
        obs = self.env.reset()
        obs = self.observe(obs)
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observe(obs)
        return obs, reward, done, info



class GraphEnv(gym.Env):
    def __init__(self, env: gym.Env, config: Dict) -> None:
        self.features = config["observation_features"]
        self.n_neighbors = config["n_neighbors"]
        self.env = env

        obs = th.tensor(self.env.reset(), dtype=th.float32)
        vehicle2d = obs2vehicle2d(obs, config["observation_features"])
        node_attr = get_node_attr(vehicle2d)
        edge_attr = get_edge_attr(vehicle2d, [[0], [0]])
        n_nodes = obs.shape[0]
        n_edges = n_nodes * config["n_neighbors"]

        self.action_space = self.env.action_space

        # 1st feature for each node/edge is reserved for valid flag
        self.observation_space = spaces.Dict(dict(
            x=spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_nodes, node_attr.shape[1]),
                dtype=float,
            ),
            x_valid=spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_nodes,),
                dtype=float,
            ),
            edge_index=spaces.Box(
                low=0,
                high=n_nodes - 1,
                shape=(2, n_edges),
                dtype=float,
            ),
            edge_attr=spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_edges, edge_attr.shape[1]),
                dtype=float,
            ),
            edge_valid=spaces.Box(
                low=0,
                high=n_nodes - 1,
                shape=(n_edges,),
                dtype=float,
            ),
        ))

    def raw2graph(self, obs: np.ndarray) -> Graph:
        obs = th.tensor(obs, dtype=th.float32)
        vehicle2d = obs2vehicle2d(obs=obs, features=self.features)
        graph = build_graph(nodes=vehicle2d, n_neighbors=self.n_neighbors)
        n_nodes = graph.x.shape[0]
        n_edges = graph.edge_attr.shape[0]

        x = th.zeros(self.observation_space["x"].shape, dtype=th.float32)
        x[:n_nodes] = graph.x
        
        x_valid = th.zeros(
            self.observation_space["x_valid"].shape,
            dtype=th.float32)
        x_valid[:n_nodes] = 1.0

        edge_index = th.zeros(
            self.observation_space["edge_index"].shape,
            dtype=th.float32)
        edge_index[:, :n_edges] = graph.edge_index

        edge_attr = th.zeros(
            self.observation_space["edge_attr"].shape,
            dtype=th.float32)
        edge_attr[:n_edges] = graph.edge_attr
        
        edge_valid = th.zeros(
            self.observation_space["edge_valid"].shape,
            dtype=th.float32)
        edge_valid[:n_edges] = 1.0

        ret = dict(
            x=x.numpy(),
            x_valid=x_valid,
            edge_index=edge_index,
            edge_attr=edge_attr.numpy(),
            edge_valid=edge_valid,
        )
        return ret

    def step(self, action: int) -> Tuple[Dict, float, bool, Optional[Dict]]:
        obs, reward, done, info = self.env.step(action)
        obs = self.raw2graph(obs)
        return obs, reward, done, info

    def reset(self) -> Dict:
        obs = self.env.reset()
        obs = self.raw2graph(obs)
        return obs
