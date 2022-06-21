from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch as th
from torch_geometric.data import Data as Graph

from graph_sdc.data import Vehicle2D, get_node_attr, get_edge_attr
from graph_sdc.util import dist_between_lines_2d


def get_trajectory_dist(
    coord: th.Tensor,
    velocity: th.Tensor,
    config: Dict,
) -> th.Tensor:
    """Get distance between estimated trajectories.

    Args:
        coord (th.Tensor): th.float, [n_nodes, n_feats=2]
        velocity (th.Tensor): [n_poinst, n_feats=2]
        config (dict): {"seconds"}

    Returns:
        th.Tensor: th.long, [n_nodes, n_nodes]
    """
    seconds = config["seconds"]
    
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
    dist = dist_between_lines_2d(lines_from, lines_to)
    dist = dist.view(n_nodes, n_nodes)
    return dist


def get_waypoint_dist(
    coord: th.Tensor,
    velocity: th.Tensor,
    config: Dict,
) -> th.Tensor:
    """Get distance between estimated waypoints.

    Args:
        coord (th.Tensor): th.float, [n_nodes, n_feats = 2]
        velocity (th.Tensor): [n_nodes, n_feats=2]
        config (dict): {"seconds", "sample_frequency", "discount_factor"}

    Returns:
        th.Tensor: th.long, [n_nodes, n_nodes]
    """
    device = coord.device
    seconds = config["seconds"]
    sample_freq = config["sample_frequency"]
    discount_factor = config["discount_factor"]
    n_nodes = coord.size(0)
    n_waypoints = int(ceil(seconds * sample_freq + 1))
    sample_interval = 1.0 / sample_freq
    # [n_waypoints,]
    timestamps = th.arange(n_waypoints, device=device) * sample_interval
    discount = th.full((n_waypoints,), discount_factor, device=device).\
        pow(timestamps)
    # [n_waypoints, n_nodes, n_feats = 2]
    waypoints = coord.view(1, n_nodes, 2) +\
        timestamps.view(-1, 1, 1) * velocity.view(1, n_nodes, 2)
    
    # [n_waypoints, n_nodes, n_nodes, n_feats = 2]
    diff = waypoints.unsqueeze(2) - waypoints.unsqueeze(1)
    # [n_waypoints, n_nodes, n_nodes]
    dist = th.linalg.vector_norm(diff, dim=-1)
    dist = dist * discount.view(-1, 1, 1)
    dist = th.min(dist, dim=0).values
    return dist


def get_euclidean_dist(coord: th.Tensor, **dummy) -> th.Tensor:
    """Get distance between estimated waypoints.

    Args:
        coord (th.Tensor): th.float, [n_nodes, n_feats = 2]

    Returns:
        th.Tensor: th.long, [n_nodes, n_nodes]
    """
    # [n_nodes, n_nodes, n_feats= 2]
    diff = coord.unsqueeze(1) - coord.unsqueeze(0)
    dist = th.linalg.vector_norm(diff, dim=-1)
    return dist


def spacetime_graph(
    coord: th.Tensor,
    velocity: th.Tensor,
    metric: str,
    config: Dict,
    n_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
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
    device = coord.device
    
    if metric == "default":
        metric_fn = get_euclidean_dist
    elif metric == "trajectory":
        metric_fn = get_trajectory_dist
    elif metric == "waypoint":
        metric_fn = get_waypoint_dist
    else:
        raise ValueError("Unrecognizable distance metric")
    
    dist = metric_fn(coord=coord, velocity=velocity, config=config)
    if not loop:
        dist.fill_diagonal_(th.inf)
    
    if radius is not None:
        dist[dist > radius] = th.inf
    
    if n_neighbors is not None:
        n_neighbors = min(n_neighbors, n_nodes - int(not loop))
        dist, indices_to = th.topk(dist, k=n_neighbors, dim=1, largest=False)
    else:
        n_neighbors = n_nodes
        indices_to = th.arange(n_neighbors, device=device).unsqueeze(0).\
            repeat(n_nodes, 1)
    
    indices_to = indices_to.view(-1)
    indices_from = th.arange(n_nodes, device=device).unsqueeze(1).\
        repeat(1, n_neighbors).view(-1) 
    indices = th.stack([indices_from, indices_to], dim=0)
    
    dist = dist.view(-1)
    indices = indices[:, th.isfinite(dist)]
    
    return indices


def build_graph(
    nodes: Vehicle2D,
    metric: str,
    vel_scale: float = 1.0,
    config: Dict = {},
    n_neighbors: Optional[int] = None,
    radius: Optional[float] = None,
    loop: bool = False,
) -> Graph:
    edge_index = spacetime_graph(
        n_neighbors=n_neighbors,
        radius=radius,
        coord=nodes.xy,
        velocity=nodes.vel_xy * vel_scale,
        metric=metric,
        config=config,
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
