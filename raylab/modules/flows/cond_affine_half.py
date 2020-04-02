"""
Reference:

Semi-Conditional Normalizing Flows for Semi-Supervised Learning
https://arxiv.org/abs/1905.00505
"""
from typing import Dict

import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from .abstract import ConditionalNormalizingFlow
from ..distributions.utils import _sum_rightmost


class DummyCond(nn.Module):
    """Dummy module outputting zeros on input."""

    @override(nn.Module)
    def forward(self, inputs, cond: Dict[str, torch.Tensor]):
        # pylint:disable=arguments-differ,unused-argument
        return torch.zeros(())


class CondAffine1DHalfFlow(ConditionalNormalizingFlow):
    """Conditional affine coupling layer."""

    def __init__(self, parity, scale_module=None, shift_module=None):
        super().__init__(event_dim=1)
        self.parity = parity
        self.scale = DummyCond() if scale_module is None else scale_module
        self.shift = DummyCond() if shift_module is None else shift_module

    @override(ConditionalNormalizingFlow)
    def _encode(self, inputs, cond: Dict[str, torch.Tensor]):
        z_0, z_1 = torch.chunk(inputs, 2, dim=-1)
        if self.parity:
            z_0, z_1 = z_1, z_0

        scale = self.scale(z_0, cond)
        shift = self.shift(z_0, cond)
        x_0 = z_0
        x_1 = (z_1 - shift) * torch.exp(-scale)
        if self.parity:
            x_0, x_1 = x_1, x_0

        out = torch.empty_like(inputs)
        out[..., ::2] = x_0
        out[..., 1::2] = x_1

        log_abs_det_jacobian = torch.cat([-scale, out * 0], dim=-1)
        return out, _sum_rightmost(log_abs_det_jacobian, self.event_dim)

    @override(ConditionalNormalizingFlow)
    def _decode(self, inputs, cond: Dict[str, torch.Tensor]):
        x_0, x_1 = inputs[..., ::2], inputs[..., 1::2]
        if self.parity:
            x_0, x_1 = x_1, x_0

        scale = self.scale(x_0, cond)
        shift = self.shift(x_0, cond)
        z_0 = x_0
        z_1 = torch.exp(scale) * x_1 + shift

        if self.parity:
            z_0, z_1 = z_1, z_0
        out = torch.cat([z_0, z_1], dim=-1)

        log_abs_det_jacobian = scale
        return out, _sum_rightmost(log_abs_det_jacobian, self.event_dim)
