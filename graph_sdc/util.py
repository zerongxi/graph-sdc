import torch as th

def cossin2radian(cos: th.Tensor, sin: th.Tensor) -> th.Tensor:
    radian = th.acos(cos)
    lower = th.argwhere(sin < 0)
    radian[lower] *= -1.0
    radian[lower] += 2 * th.pi
    return radian
