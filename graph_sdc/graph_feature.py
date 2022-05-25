import logging
from re import X
from typing import Dict, Tuple
from gym import spaces
import torch as th
from torch.nn import functional as F

from torch_geometric.nn import Linear, GATv2Conv, TransformerConv
from torch_geometric.data import Data as Graph
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from graph_sdc.graph_util import combine_graphs


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        config: Dict,
        graph_cls_name: str,
    ):
        features_dim = config["node_dims"][-1]
        if config.get("concat_heads", True):
            n_heads = config.get("n_heads", 1)
            if isinstance(n_heads, int):
                n_heads = [n_heads]
            features_dim *= n_heads[-1]
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim,
        )
        self.config = config
        self.negative_slope = self.config.get("negative_slope", 0.01)
        self.graph_cls_name = graph_cls_name
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
                node_embedding_layers.append(th.nn.LeakyReLU(self.negative_slope))
                edge_embedding_layers.append(Linear(-1, out_dim))
                edge_embedding_layers.append(th.nn.LeakyReLU(self.negative_slope))
        self.node_embedding_net = th.nn.Sequential(*node_embedding_layers)\
            if len(node_embedding_layers) > 0\
            else lambda x: x
        self.edge_embedding_net = th.nn.Sequential(*edge_embedding_layers)\
            if len(edge_embedding_layers) > 0\
            else lambda x: x

    def _build_graph_net(self):
        node_dims = list(self.config["node_dims"])
        edge_dims = list(self.config["edge_dims"])
        n_heads = self.config.get("n_heads", 1)
        if isinstance(n_heads, int):
            n_heads = [n_heads] * len(node_dims)
        concat_heads = self.config.get("concat_heads", True)
        self.graph_layers = []
        self.edge_layers = []

        for idx, _ in enumerate(node_dims):
            if self.graph_cls_name == "xfmr":
                logging.info("Build Graph Transformer Layer")
                self.graph_layers.append(TransformerConv(
                    in_channels=-1,
                    out_channels=node_dims[idx],
                    heads=n_heads[idx],
                    concat=concat_heads,
                    edge_dim=-1,
                ))
            elif self.graph_cls_name == "gat":
                logging.info("Build Graph Attention Layer")
                self.graph_layers.append(GATv2Conv(
                    in_channels=-1,
                    out_channels=node_dims[idx],
                    heads=n_heads[idx],
                    concat=concat_heads,
                    edge_dim=-1,
                    add_self_loops=True,
                    fill_value=0.0,
                    negative_slope=0.2,
                ))
            self.add_module("graph_conv{}".format(idx), self.graph_layers[-1])
            self.edge_layers.append(Linear(-1, edge_dims[idx]))
            self.add_module("edge_linear{}".format(idx), self.edge_layers[-1])

    def preprocessing(self, data: Dict) -> Tuple[Graph, th.Tensor, th.Tensor]:
        data["edge_index"] = data["edge_index"].long()
        # remove invalid nodes/edges
        graph = []
        for x, x_valid, edge_index, edge_attr, edge_valid in zip(
                data["x"], data["x_valid"], data["edge_index"],
                data["edge_attr"], data["edge_valid"]):
            graph.append(Graph(
                x=x[x_valid.bool()],
                edge_index=edge_index[:, edge_valid.bool()],
                edge_attr=edge_attr[edge_valid.bool()],
            ))
        graph, n_nodes, node_index_shift = combine_graphs(graph)
        n_nodes = th.tensor(n_nodes, dtype=th.long)
        node_index_shift = th.tensor(node_index_shift, dtype=th.long)
        return graph, n_nodes, node_index_shift

    # TODO: update pooling method
    def forward(self, data: Dict) -> th.Tensor:
        graph, n_nodes, node_index_shift = self.preprocessing(data)

        # embedding
        x = self.node_embedding_net(graph.x)
        edge_index = graph.edge_index
        edge_attr = self.edge_embedding_net(graph.edge_attr)

        # conv
        for graph_l, edge_l in zip(self.graph_layers, self.edge_layers):
            # edge_attr = th.cat(
            #     [edge_attr, x[edge_index[0]], x[edge_index[1]]], dim=-1)
            edge_attr = edge_l(edge_attr)
            edge_attr = F.leaky_relu(
                edge_attr, negative_slope=self.negative_slope)
            x = graph_l(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)

        x = x[node_index_shift]
        return x
