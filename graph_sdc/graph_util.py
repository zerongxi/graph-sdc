from typing import List, Sequence, Tuple
import numpy as np
import torch as th
from torch_geometric.data import Data as Graph
from torch_geometric.nn import knn_graph

from graph_sdc.data import Vehicle2D, get_node_attr, get_edge_attr


def build_graph(
    nodes: Vehicle2D,
    n_neighbors: int,
) -> Graph:
    edge_index = knn_graph(nodes.xy, k=n_neighbors, loop=False)
    
    node_attr = get_node_attr(nodes)
    edge_attr = get_edge_attr(nodes, edge_index)
    
    graph_data = Graph(
        x=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return graph_data


def combine_graphs(
    graphs: Sequence[Graph]
) -> Tuple[Graph, List[int], List[int]]:
    x = [g.x for g in graphs]
    edge_index = [g.edge_index for g in graphs]
    
    n_nodes = [u.shape[0] for u in x]
    node_index_shift = np.add.accumulate([0] + n_nodes[:-1])
    edge_index = [e + s for e, s in zip(edge_index, node_index_shift)]
    
    x = th.cat(x, dim=0)
    edge_index = th.cat(edge_index, dim=1)
    
    edge_attr = None
    if graphs[0].edge_index is not None:
        edge_attr = [g.edge_attr for g in graphs]
        edge_attr = th.cat(edge_attr, dim=0)
    combined_graph = Graph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return combined_graph, n_nodes, node_index_shift
