"""
Reference:

Semi-Conditional Normalizing Flows for Semi-Supervised Learning
https://arxiv.org/abs/1905.00505
"""
import torch
from ray.rllib.utils.annotations import override

from raylab.modules.basic.lambd import Lambda
from .abstract import NormalizingFlow


class CondAffineHalfFlow(NormalizingFlow):
    """Conditional affine coupling layer."""

    def __init__(self, parity, scale_module=None, shift_module=None):
        super().__init__()
        self.parity = parity
        if scale_module is None:
            self.s_cond = Lambda(lambda _: torch.zeros([]))
        else:
            self.s_cond = scale_module
        if shift_module is None:
            self.t_cond = Lambda(lambda _: torch.zeros([]))
        else:
            self.t_cond = shift_module

    @override(NormalizingFlow)
    def _encode(self, inputs):
        var, cond = inputs
        x_0, x_1 = var[..., ::2], var[..., 1::2]
        if self.parity:
            x_0, x_1 = x_1, x_0
        scale = self.s_cond([x_0, cond])
        shift = self.t_cond([x_0, cond])
        z_0 = x_0  # untouched half
        z_1 = torch.exp(scale) * x_1 + shift
        if self.parity:
            z_0, z_1 = z_1, z_0
        out = torch.cat([z_0, z_1], dim=-1)
        log_det = torch.sum(scale, dim=-1)
        return out, log_det

    @override(NormalizingFlow)
    def _decode(self, inputs):
        var, cond = inputs
        z_0, z_1 = torch.chunk(var, 2, dim=-1)
        if self.parity:
            z_0, z_1 = z_1, z_0
        scale = self.s_cond([z_0, cond])
        shift = self.t_cond([z_0, cond])
        x_0 = z_0
        x_1 = (z_1 - shift) * torch.exp(-scale)
        if self.parity:
            x_0, x_1 = x_1, x_0
        mask0 = torch.zeros_like(var).bool()
        mask0[..., ::2] = True
        mask1 = ~mask0
        out = torch.empty_like(var)
        out[mask0] = x_0.flatten()
        out[mask1] = x_1.flatten()
        log_det = torch.sum(-scale, dim=-1)
        return out, log_det
