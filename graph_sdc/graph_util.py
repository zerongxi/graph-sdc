from typing import List, Optional, Sequence, Tuple
import numpy as np
import torch as th
from torch_geometric.data import Data as Graph
from torch_geometric.nn import knn_graph

from graph_sdc.data import Vehicle2D, get_node_attr, get_edge_attr
from graph_sdc.util import dist_between_lines_2d


def get_trajectory_dist(
    coord: th.Tensor,
    velocity: th.Tensor,
    seconds: float,
) -> th.Tensor:
    """Get distance between estimated trajectories.

    Args:
        coord (th.Tensor): th.float, [n_nodes, n_feats=2]
        velocity (th.Tensor): [n_poinst, n_feats=2]
        seconds: seconds looking ahead

    Returns:
        th.Tensor: th.long, [n_nodes, n_nodes]
    """
    n_nodes = coord.size(0)
    # [n_nodes, n_feats = 2]
    start_p = coord
    end_p = coord + velocity * seconds
    # [2, n_feats = 2, n_nodes]
    lines = th.stack([start_p.T, end_p.T], dim=0)
    # [2, n_feats = 2, n_lines = n_nodes x n_nodes]
    lines_from = lines.unsqueeze(-1).repeat(1, 1, 1, n_nodes).\
        view(2, 2, n_nodes * n_nodes)
    lines_to = lines.unsqueeze(-2).repeat(1, 1, n_nodes, 1).\
        view(2, 2, n_nodes * n_nodes)
    #print(th.stack([lines_from, lines_to], dim=-1).permute((2, 3, 0, 1)))
    dist = dist_between_lines_2d(lines_from, lines_to)
    dist = dist.view(n_nodes, n_nodes)
    return dist


def get_waypoint_dist(
    coord: th.Tensor,
    velocity: th.Tensor,
    seconds: float,
    n_waypoints: int,
) -> th.Tensor:
    """Get distance between estimated waypoints.

    Args:
        coord (th.Tensor): th.float, [n_nodes, n_feats = 2]
        velocity (th.Tensor): [n_nodes, n_feats=2]
        seconds: seconds looking ahead
        n_waypoints: num of waypoints looking ahead

    Returns:
        th.Tensor: th.long, [n_nodes, n_nodes]
    """
    print("waypoint")
    n_nodes = coord.size(0)
    timestamps = th.linspace(
        0.0, seconds, n_waypoints + 1, device=coord.device
    ).view(-1, 1, 1)
    # [n_waypoints, n_nodes, n_feats = 2]
    waypoints = coord.view(1, n_nodes, 2) +\
        timestamps * velocity.view(1, n_nodes, 2)
    
    # [n_waypoints, n_nodes, n_nodes, n_feats = 2]
    diff = waypoints.unsqueeze(2) - waypoints.unsqueeze(1)
    dist = th.linalg.vector_norm(diff, dim=-1)
    dist = th.min(dist, dim=0).values
    return dist


def spacetime_knn_graph(
    n_neighbors: int,
    coord: th.Tensor,
    velocity: th.Tensor,
    metric: str,
    seconds: float,
    n_waypoints: Optional[int] = None,
    loop: bool = False,
) -> th.Tensor:
    """Build knn graph for self-driving scenario.

    Args:
        n_neighbors (int): num of neighbors
        coord (th.Tensor): th.float, [n_nodes, n_feats=2]
        velocity (th.Tensor): [n_poinst, n_feats=2]
        seconds: seconds looking ahead

    Returns:
        th.Tensor: th.long, [n_dims=2, n_edges]
    """
    n_nodes = coord.size(0)
    n_neighbors = min(n_neighbors, n_nodes - int(not loop))
    if metric == "trajectory":
        dist = get_trajectory_dist(
            coord=coord,
            velocity=velocity,
            seconds=seconds)
    elif metric == "waypoint":
        dist = get_waypoint_dist(
            coord=coord,
            velocity=velocity,
            seconds=seconds,
            n_waypoints=n_waypoints)
    
    if not loop:
        dist.fill_diagonal_(th.inf)
    
    topk = th.topk(dist, k=n_neighbors, dim=1, largest=False).indices
    indices_to = topk.view(-1)
    indices_from = th.arange(
        0, n_nodes, dtype=th.long, device=indices_to.device
    ).unsqueeze(1).repeat(1, n_neighbors).view(-1)
    
    return th.stack([indices_from, indices_to], dim=0)


def build_graph(
    nodes: Vehicle2D,
    n_neighbors: int,
    metric: str,
    vel_scale: float = 1.0,
    seconds: Optional[float] = None,
    n_waypoints: Optional[int] = None,
    loop: bool = False,
) -> Graph:
    if metric == "default":
        edge_index = knn_graph(nodes.xy, k=n_neighbors, loop=False)
    else:
        edge_index = spacetime_knn_graph(
            n_neighbors=n_neighbors,
            coord=nodes.xy,
            velocity=nodes.vel_xy * vel_scale,
            seconds=seconds,
            metric=metric,
            n_waypoints=n_waypoints,
            loop=loop)
    
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
