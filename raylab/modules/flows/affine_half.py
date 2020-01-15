"""
Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)
"""
import torch
from ray.rllib.utils.annotations import override

from raylab.modules.lambd import Lambda
from .abstract import NormalizingFlow


class AffineHalfFlow(NormalizingFlow):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of
    the dimensions in x are linearly scaled/transformed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, parity, scale_module=None, shift_module=None):
        super().__init__()
        self.parity = parity
        if scale_module is None:
            self.s_cond = Lambda(torch.zeros_like)
        else:
            self.s_cond = scale_module
        if shift_module is None:
            self.t_cond = Lambda(torch.zeros_like)
        else:
            self.t_cond = shift_module

    @override(NormalizingFlow)
    def _encode(self, inputs):
        x_0, x_1 = inputs[..., ::2], inputs[..., 1::2]
        if self.parity:
            x_0, x_1 = x_1, x_0
        scale = self.s_cond(x_0)
        shift = self.t_cond(x_0)
        z_0 = x_0
        z_1 = torch.exp(scale) * x_1 + shift
        if self.parity:
            z_0, z_1 = z_1, z_0
        out = torch.cat([z_0, z_1], dim=-1)
        log_det = torch.sum(scale, dim=-1)
        return out, log_det

    @override(NormalizingFlow)
    def _decode(self, inputs):
        z_0, z_1 = inputs[..., ::2], inputs[..., 1::2]
        if self.parity:
            z_0, z_1 = z_1, z_0
        scale = self.s_cond(z_0)
        shift = self.t_cond(z_0)
        x_0 = z_0
        x_1 = (z_1 - shift) * torch.exp(-scale)
        if self.parity:
            x_0, x_1 = x_1, x_0
        out = torch.cat([x_0, x_1], dim=-1)
        log_det = torch.sum(-scale, dim=-1)
        return out, log_det
