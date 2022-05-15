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