from typing import Dict, Optional, Tuple
import gym
from gym import spaces
import numpy as np
import torch as th

from torch_geometric.data import Data as Graph

from graph_sdc.data import Vehicle2D, obs2vehicle2d, get_node_attr, get_edge_attr
from graph_sdc.graph_util import build_graph


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
                shape=(n_nodes, node_attr.shape[1] + 1),
                dtype=np.float32,
            ),
            edge_index=spaces.Box(
                low=0,
                high=n_nodes - 1,
                shape=(2 + 1, n_edges),
                dtype=np.float32,
            ),
            edge_attr=spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_edges, edge_attr.shape[1] + 1),
                dtype=np.float32,
            )
        ))
        
    
    def raw2graph(self, obs: np.ndarray) -> Graph:
        obs = th.tensor(obs, dtype=th.float32)
        vehicle2d = obs2vehicle2d(obs=obs, features=self.features)
        graph = build_graph(nodes=vehicle2d, n_neighbors=self.n_neighbors)
        n_nodes = graph.x.shape[0]
        n_edges = graph.edge_attr.shape[0]
        
        x = th.zeros(self.observation_space["x"].shape, dtype=th.float32)
        x[:n_nodes, 0] = 1
        x[:n_nodes, 1:] = graph.x
        
        edge_index = th.zeros(
            self.observation_space["edge_index"].shape,
            dtype=th.float32,
        )
        edge_index[0, :n_edges] = 1
        edge_index[1:, :n_edges] = graph.edge_index
        
        edge_attr = th.zeros(
            self.observation_space["edge_attr"].shape,
            dtype=th.float32
        )
        edge_attr[:n_edges, 0] = 1
        edge_attr[:n_edges, 1:] = graph.edge_attr
        
        return dict(
            x=x,
            edge_index = edge_index,
            edge_attr=edge_attr,
        )
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Optional[Dict]]:
        obs, reward, done, info = self.env.step(action)
        obs = self.raw2graph(obs)
        return obs, reward, done, info
    
    def reset(self) -> Dict:
        obs = self.env.reset()
        obs = self.raw2graph(obs)
        return obs
