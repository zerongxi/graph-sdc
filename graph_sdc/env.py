from copy import deepcopy
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


class HighwayEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config: Dict):
        logging.info("Using HighwayEnvWrapper")
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
        
        config["observation"]["normalize"] = False
        env.configure(config)
        env.reset()
        super().__init__(env)

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
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        obs_dict = dict(zip(self.features, obs.T))
        obs_dict = self.shift_obs(obs_dict)
        if self.normalize:
            obs_dict = self.normalize_obs(obs_dict)
        obs = [obs_dict[k] for k in self.features]
        obs = np.stack(obs, axis=1)
        return obs


class GraphEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, config:Dict):
        logging.info("Using GraphEnvWrapper")
        super().__init__(env)
        self.config = config
        self.features = config["observation_features"]
        self.n_neighbors = config["n_neighbors"]
        vx_range = config["observation_features_range"]["vx"]
        x_range = config["observation_features_range"]["x"]
        self.vel_scale = (vx_range[1] - vx_range[0]) /\
            (x_range[1] - x_range[0])

        obs = th.tensor(self.env.reset(), dtype=th.float32)
        vehicle2d = obs2vehicle2d(obs, config["observation_features"])
        node_attr = get_node_attr(vehicle2d)
        edge_attr = get_edge_attr(vehicle2d, [[0], [0]])
        n_nodes = obs.shape[0]
        n_edges = n_nodes * config["n_neighbors"]
        
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

    def observation(self, obs: np.ndarray) -> Graph:
        obs = th.tensor(obs, dtype=th.float32)
        vehicle2d = obs2vehicle2d(
            obs=obs, features=self.config["observation_features"])
        graph_metric = self.config.get("metric", "default")
        graph = build_graph(
            nodes=vehicle2d,
            n_neighbors=self.n_neighbors,
            metric=graph_metric,
            vel_scale=self.vel_scale,
            config=self.config.get(graph_metric, {}))
        n_nodes = graph.x.size(0)
        n_edges = graph.edge_attr.size(0)

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
            x=x,
            x_valid=x_valid,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_valid=edge_valid,
        )
        return ret
