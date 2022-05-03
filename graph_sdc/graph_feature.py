from typing import Dict, Optional
from gym import spaces
import torch as th

from torch_geometric.nn import GATConv, Sequential
from torch_geometric.data import Data as Graph
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from graph_sdc.graph_util import combine_graphs


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        config: Dict,
        features_dim: Optional[int]
    ):
        if features_dim is None:
            features_dim = config["node_dims"][-1]
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
        )
        self.config = config
        self._build_net()

    def _build_net(self):
        raise NotImplementedError

    def forward(self, data: Dict) -> th.Tensor:
        raise NotImplementedError


class GATFeaturesExtractor(GraphFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, config: Dict):
        features_dim = config["node_dims"][-1]
        if config.get("concat_heads", True):
            n_heads = config.get("n_heads", 1)
            if isinstance(n_heads, int):
                n_heads = [n_heads]
            features_dim *= n_heads[-1]
        super().__init__(observation_space, config, features_dim)

    # TODO: update edge_dim
    def _build_net(self):
        node_dims = list(self.config["node_dims"])
        n_layers = len(node_dims)
        n_heads = self.config.get("n_heads", 1)
        concat_heads = self.config.get("concat_heads", True)
        if isinstance(n_heads, int):
            n_heads = [n_heads] * n_layers
        layers = []
        for i in range(n_layers):
            layers.append((GATConv(
                in_channels=-1,
                out_channels=node_dims[i],
                heads=n_heads[i],
                edge_dim=-1,
                fill_value=0.0,
                add_self_loops=True,
                concat=concat_heads,
            ), "x, edge_index, edge_attr -> x"))
            layers.append(th.nn.ReLU())
        self.net = Sequential("x, edge_index, edge_attr", layers)

    #TODO: update pooling method at last
    def forward(self, data: Dict) -> th.Tensor:
        data["edge_index"] = data["edge_index"].long()
        
        # remove invalid nodes/edges
        graph = []
        for x, edge_index, edge_attr in\
                zip(data["x"], data["edge_index"], data["edge_attr"]):
            x_valid = x[:, 0].bool()
            edge_index_valid = edge_index[0, :].bool()
            edge_attr_valid = edge_attr[:, 0].bool()
            graph.append(Graph(
                x=x[x_valid, 1:],
                edge_index=edge_index[1:, edge_index_valid],
                edge_attr=edge_attr[edge_attr_valid, :],
            ))
        graph, n_nodes, node_index_shift = combine_graphs(graph)
        del data
        
        logits = self.net(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
        )
        
        index = th.tensor(node_index_shift, dtype=th.long)
        logits = logits[index]
        return logits
