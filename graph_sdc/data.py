from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np

import torch as th

from graph_sdc.util import cossin2radian


@dataclass
class Vehicle2D:
    # [n_vehicles, 2], th.float32
    xy: th.Tensor
    # [n_vehicles, 2], th.float32
    vel_xy: th.Tensor
    # radius angle
    # [n_vehicles, 1], th.float32
    heading: Optional[th.Tensor] = None
    # [n_vehicles, 2], th.float32
    heading_xy: Optional[th.Tensor] = None


def obs2vehicle2d(
    obs: th.Tensor,
    features: Sequence[str]
) -> Vehicle2D:
    """_summary_

    Args:
        obs (th.Tensor): [n_vehicles, n_features]
        features (Sequence[str]): Names of features.

    Returns:
        Vehicle2D
    """
    # remove unpresented vehicles
    presence = obs[:, features.index("presence")].bool()
    obs = obs[presence]

    # [n_vehicles, n_features] -> feature: [n_vehicles,]
    obs = {u: v for u, v in zip(features, obs.T)}

    xy = th.stack([obs["x"], obs["y"]], dim=1)
    vel_xy = th.stack([obs["vx"], obs["vy"]], dim=1)

    heading = None
    if "heading" in obs:
        heading = obs["heading"].view(-1, 1)

    heading_xy = None
    if "cos_h" in obs and "sin_h" in obs:
        heading_xy = th.stack([obs["cos_h"], obs["sin_h"]], dim=1)

    vehicle2d = Vehicle2D(
        xy=xy,
        vel_xy=vel_xy,
        heading=heading,
        heading_xy=heading_xy,
    )
    return vehicle2d


def get_node_attr(
    nodes: Vehicle2D,
    sdc_relative: bool = False,
    sdc_index: int = 0,
) -> th.Tensor:
    features = []

    features.append(nodes.xy)
    features.append(nodes.vel_xy)

    heading_xy = None
    if nodes.heading_xy is not None:
        heading_xy = nodes.heading_xy
        features.append(heading_xy)
    elif nodes.heading is not None:
        cos_h = th.cos(nodes.heading)
        sin_h = th.sin(nodes.heading)
        heading_xy = th.stack([cos_h, sin_h], dim=1)
        features.append(heading_xy)
    
    if sdc_relative:
        features.append(nodes.xy - nodes.xy[sdc_index])
        features.append(nodes.vel_xy - nodes.vel_xy[sdc_index])
        #TODO: add relative heading_xy

    features_concat = th.cat(features, dim=-1)
    return features_concat


def get_edge_attr(
    nodes: Vehicle2D,
    edge_index: th.Tensor,
) -> th.Tensor:
    """_summary_

    Args:
        nodes (Vehicle2D): Vehicle2D
        edge_index (th.Tensor): [2, n_edges]

    Returns:
        th.Tensor: [n_edges, n_features]
    """
    features = []

    xy_diff = nodes.xy[edge_index[0]] - nodes.xy[edge_index[1]]
    features.append(xy_diff)

    distance = th.sqrt(th.sum(th.square(xy_diff), dim=1)).view(-1, 1)
    features.append(distance)

    vel_xy_diff = nodes.vel_xy[edge_index[0]] - nodes.vel_xy[edge_index[1]]
    features.append(vel_xy_diff)

    # TODO: calculate heading_xy with triangular formulas
    heading = None
    if nodes.heading is not None:
        heading = nodes.heading[:, 0]
    elif nodes.heading_xy is not None:
        heading = cossin2radian(nodes.heading_xy[:, 0], nodes.heading_xy[:, 1])

    if heading is not None:
        heading_diff = heading[edge_index[0]] - heading[edge_index[1]]
        cos_h = th.cos(heading_diff)
        sin_h = th.sin(heading_diff)
        heading_diff_xy = th.stack([cos_h, sin_h], dim=1)
        features.append(heading_diff_xy)

    features_concat = th.cat(features, dim=1)
    return features_concat
