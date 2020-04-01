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

from raylab.modules.basic.lambd import Lambda
from .abstract import NormalizingFlow


class Affine1DHalfFlow(NormalizingFlow):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of
    the elements in the last dimension of x are linearly scaled/transformed as a
    function of the other half. Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """

    def __init__(self, parity, scale_module=None, shift_module=None, **kwargs):
        super().__init__(event_dim=1, **kwargs)
        self.parity = parity
        if scale_module is None:
            self.scale = Lambda(lambda _: torch.zeros([]))
        else:
            self.scale = scale_module
        if shift_module is None:
            self.shift = Lambda(lambda _: torch.zeros([]))
        else:
            self.shift = shift_module

    @override(NormalizingFlow)
    def _encode(self, inputs):
        z_0, z_1 = torch.chunk(inputs, 2, dim=-1)
        if self.parity:
            z_0, z_1 = z_1, z_0

        scale = self.scale(z_0)
        shift = self.shift(z_0)
        x_0 = z_0
        x_1 = (z_1 - shift) * torch.exp(-scale)

        if self.parity:
            x_0, x_1 = x_1, x_0
        mask0 = torch.zeros_like(inputs).bool()
        mask0[..., ::2] = True
        mask1 = ~mask0
        out = torch.empty_like(inputs)
        out[mask0] = x_0.flatten()
        out[mask1] = x_1.flatten()

        log_abs_det_jacobian = -scale
        return out, log_abs_det_jacobian

    @override(NormalizingFlow)
    def _decode(self, inputs):
        x_0, x_1 = inputs[..., ::2], inputs[..., 1::2]
        if self.parity:
            x_0, x_1 = x_1, x_0

        scale = self.scale(x_0)
        shift = self.shift(x_0)
        z_0 = x_0
        z_1 = torch.exp(scale) * x_1 + shift

        if self.parity:
            z_0, z_1 = z_1, z_0
        out = torch.cat([z_0, z_1], dim=-1)

        log_abs_det_jacobian = scale
        return out, log_abs_det_jacobian
