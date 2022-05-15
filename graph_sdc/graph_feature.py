import logging
from re import X
from typing import Dict, Optional, Tuple
from gym import spaces
import torch as th

from torch_geometric.nn import Sequential, Linear, GATv2Conv, TransformerConv
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
        self._build_embedding_net()
        self._build_graph_net()

    def _build_embedding_net(self):
        embedding_dims = self.config.get("embedding_dims", None)
        node_embedding_layers = []
        edge_embedding_layers = []
        if embedding_dims is not None:
            logging.info("Building Embedding Network")
            for out_dim in embedding_dims:
                node_embedding_layers.append(Linear(-1, out_dim))
                node_embedding_layers.append(th.nn.ReLU())
                edge_embedding_layers.append(Linear(-1, out_dim))
                edge_embedding_layers.append(th.nn.ReLU())
        self.node_embedding_net = th.nn.Sequential(*node_embedding_layers)\
            if len(node_embedding_layers) > 0\
            else lambda x: x
        self.edge_embedding_net = th.nn.Sequential(*edge_embedding_layers)\
            if len(edge_embedding_layers) > 0\
            else lambda x: x

    def _build_graph_net(self):
        raise NotImplementedError
    
    def preprocessing(self, data: Dict) -> Tuple[Graph, th.Tensor, th.Tensor]:
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
                edge_attr=edge_attr[edge_attr_valid, 1:],
            ))
        graph, n_nodes, node_index_shift = combine_graphs(graph)
        n_nodes = th.tensor(n_nodes, dtype=th.long)
        node_index_shift = th.tensor(node_index_shift, dtype=th.long)
        return graph, n_nodes, node_index_shift        

    #TODO: update pooling method
    def forward(self, data: Dict) -> th.Tensor:
        graph, n_nodes, node_index_shift = self.preprocessing(data)
        
        logits = self.graph_net(
            x=self.node_embedding_net(graph.x),
            edge_index=graph.edge_index,
            edge_attr=self.edge_embedding_net(graph.edge_attr),
        )
        
        logits = logits[node_index_shift]
        return logits


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
    def _build_graph_net(self):
        logging.info("Building Graph Attention Networks")
        node_dims = list(self.config["node_dims"])
        n_heads = self.config.get("n_heads", 1)
        if isinstance(n_heads, int):
            n_heads = [n_heads] * len(node_dims)
        concat_heads = self.config.get("concat_heads", True)
        graph_layers = []
        for n_dim, heads in zip(node_dims, n_heads):
            graph_layers.append((GATv2Conv(
                in_channels=-1,
                out_channels=n_dim,
                heads=heads,
                edge_dim=-1,
                fill_value=0.0,
                add_self_loops=True,
                concat=concat_heads,
            ), "x, edge_index, edge_attr -> x"))
            graph_layers.append((th.nn.LeakyReLU(), "x -> x"))
        self.graph_net = Sequential("x, edge_index, edge_attr", graph_layers)


class TransformerFeaturesExtractor(GraphFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, config: Dict):
        features_dim = config["node_dims"][-1]
        if config.get("concat_heads", True):
            n_heads = config.get("n_heads", 1)
            if isinstance(n_heads, int):
                n_heads = [n_heads]
            features_dim *= n_heads[-1]
        super().__init__(observation_space, config, features_dim)

    # TODO: update edge_dim
    def _build_graph_net(self):
        logging.info("Building Graph Transformer Networks")

        node_dims = list(self.config["node_dims"])
        n_heads = self.config.get("n_heads", 1)
        if isinstance(n_heads, int):
            n_heads = [n_heads] * len(node_dims)
        concat_heads = self.config.get("concat_heads", True)
        graph_layers = []

        for n_dim, heads in zip(node_dims, n_heads):
            graph_layers.append((TransformerConv(
                in_channels=-1,
                out_channels=n_dim,
                heads=heads,
                edge_dim=-1,
                concat=concat_heads,
            ), "x, edge_index, edge_attr -> x"))
            graph_layers.append((th.nn.LeakyReLU(), "x -> x"))
        self.graph_net = Sequential("x, edge_index, edge_attr", graph_layers)
