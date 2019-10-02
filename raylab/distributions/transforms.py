"""Collection of invertible transform functions, or Flows."""
import torch
from torch.distributions import constraints
from torch.distributions.transforms import Transform


class TanhTransform(Transform):
    """Transform via the mapping :math:`y = \frac{e^x - e^{-x}} {e^x + e^{-x}}`."""

    domain = constraints.real
    codomain = constraints.interval(-1, +1)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y):
        to_log1 = 1 + y
        to_log2 = 1 - y
        to_log1[to_log1 == 0] += torch.finfo(y.dtype).eps
        to_log2[to_log2 == 0] += torch.finfo(y.dtype).eps
        return (torch.log(to_log1) - torch.log(to_log2)) / 2

    def log_abs_det_jacobian(self, x, y):
        to_log = 1 - y.pow(2)
        to_log[to_log == 0] += torch.finfo(y.dtype).eps
        return torch.log(to_log)
