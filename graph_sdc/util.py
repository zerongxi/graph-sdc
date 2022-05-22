from typing import Tuple
import torch as th

def cossin2radian(cos: th.Tensor, sin: th.Tensor) -> th.Tensor:
    radian = th.acos(cos)
    lower = th.argwhere(sin < 0)
    radian[lower] *= -1.0
    radian[lower] += 2 * th.pi
    return radian


def trigonometic_diff(
    cos1: float,
    sin1: float,
    cos2: float,
    sin2: float,
) -> Tuple[float, float]:
    cos_ = cos1 * cos2 + sin1 * sin2
    sin_ = sin1 * cos2 - cos1 * sin2
    return cos_, sin_


def linear_map(
    v: float,
    x: Tuple[float, float],
    y: Tuple[float, float]
) -> float:
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


def check_line_intersect_2d(line1: th.Tensor, line2: th.Tensor) -> th.Tensor:
    """check if line1 and line2 intersect.

    Args:
        line1 (th.Tensor): th.float, [n_points = 2, n_feats = 2, n_lines]
        line2 (th.Tensor): th.float, [n_points = 2, n_feats = 2, n_lines]

    Returns:
        th.Tensor: th.bool, [n_lines,]
    """
    def check_orientation(_a, _b, _c):
        val = (_b[1] - _a[1]) * (_c[0] - _a[0]) -\
            (_c[1] - _a[1]) * (_b[0] - _a[0])
        return th.sign(val).long()
    
    def check_on_segment(_a, _b, _c):
        """Check if _a is on the segment (_b, _c)."""
        return th.logical_and(
            th.logical_and(
                _a[0] <= th.maximum(_b[0], _c[0]),
                _a[0] >= th.minimum(_b[0], _c[0])),
            th.logical_and(
                _a[1] <= th.maximum(_b[1], _c[1]),
                _a[1] >= th.minimum(_b[1], _c[1])))

    a = line1[0]
    b = line1[1]
    c = line2[0]
    d = line2[1]
    
    o1 = check_orientation(a, b, c)
    o2 = check_orientation(a, b, d)
    o3 = check_orientation(c, d, a)
    o4 = check_orientation(c, d, b)
    
    intersect =[
        th.logical_and(o1 != o2, o3 != o4),
        th.logical_and(o1 == 0, check_on_segment(c, a, b)),
        th.logical_and(o2 == 0, check_on_segment(d, a, b)),
        th.logical_and(o3 == 0, check_on_segment(a, c, d)),
        th.logical_and(o4 == 0, check_on_segment(b, c, d))]
    return th.any(th.stack(intersect, dim=0), dim=0)


def dist_point_to_line_2d(points: th.Tensor, lines: th.Tensor) -> th.Tensor:
    """Compute the distance between a point and a line
    
    Args:
        point (th.Tensor): th.float, [n_feats = 2, n_lines]
        line (th.Tensor): th.float, [n_points = 2, n_feats = 2, n_lines]

    Returns:
        th.Tensor: th.float, [n_lines,]
    """
    lines_vec = lines[1] - lines[0]
    points_vec = points - lines[0]
    lines_len = th.linalg.vector_norm(lines_vec, dim=0, keepdims=True)
    lines_unit_vec = lines_vec / lines_len
    points_scaled_vec = points_vec / lines_len
    t = lines_unit_vec[0] * points_scaled_vec[0] +\
        lines_unit_vec[1] * points_scaled_vec[1]
    t = th.clip(t, 0.0, 1.0).unsqueeze(0)
    nearest = lines_vec * t
    dist = th.linalg.vector_norm(points_vec - nearest, dim=0)
    return dist


def dist_between_lines_2d(line1: th.Tensor, line2: th.Tensor) -> th.Tensor:
    """Compute distance between line segments

    Args:
        line1 (th.Tensor): th.float, [n_points = 2, n_feats=2, n_lines]
        line2 (th.Tensor): th.float, [n_points = 2, n_feats=2, n_lines]

    Returns:
        th.Tensor: th.float, [n_lines,]
    """
    n_lines = line1.size(-1)
    
    intersect = check_line_intersect_2d(line1, line2)
    non_intersect = th.logical_not(intersect)
    
    non_intersect_line1 = line1[:, :, non_intersect]
    non_intersect_line2 = line2[:, :, non_intersect]
    non_intersect_dist = [
        dist_point_to_line_2d(non_intersect_line1[0], non_intersect_line2),
        dist_point_to_line_2d(non_intersect_line1[1], non_intersect_line2),
        dist_point_to_line_2d(non_intersect_line2[0], non_intersect_line1),
        dist_point_to_line_2d(non_intersect_line2[1], non_intersect_line1)]
    non_intersect_dist = th.stack(non_intersect_dist, dim=0)
    non_intersect_dist = th.min(non_intersect_dist, dim=0).values
    
    dist = th.zeros(
        (n_lines,), dtype=th.float, device=non_intersect_dist.device)
    dist[non_intersect] = non_intersect_dist
    return dist
