"""Distribution transforms as PyTorch modules compatible with TorchScript."""
# pylint:disable=missing-docstring
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils.annotations import override

from .utils import _sum_rightmost


class Transform(nn.Module):
    def __init__(self, event_dim=0):
        super().__init__()
        self.event_dim = event_dim

    @override(nn.Module)
    def forward(self, inputs, reverse: bool = False):
        # pylint:disable=arguments-differ,assignment-from-no-return
        if reverse:
            out = self._decode(inputs)
            log_abs_det_jacobian = -self.log_abs_det_jacobian(out, inputs)
        else:
            out = self._encode(inputs)
            log_abs_det_jacobian = self.log_abs_det_jacobian(inputs, out)
        return out, _sum_rightmost(log_abs_det_jacobian, self.event_dim)

    def _encode(self, inputs):
        """
        Computes the transform `x => y`.
        """

    def _decode(self, inputs):
        """
        Inverts the transform `y => x`.
        """

    def log_abs_det_jacobian(self, inputs, outputs):
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        """


class InvTransform(Transform):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    @override(Transform)
    def forward(self, inputs, reverse: bool = False):
        # pylint:disable=protected-access
        return self.transform(inputs, reverse=not reverse)


class ComposeTransform(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.inv_transforms = nn.ModuleList(transforms[::-1])

    @override(nn.Module)
    def forward(self, inputs, reverse: bool = False):  # pylint:disable=arguments-differ
        out = inputs
        if reverse:
            log_abs_det_jacobian = 0.0
            for transform in self.inv_transforms:
                out, log_det = transform(out, reverse=reverse)
                log_abs_det_jacobian = log_abs_det_jacobian + log_det
        else:
            log_abs_det_jacobian = 0.0
            for transform in self.transforms:
                out, log_det = transform(out, reverse=reverse)
                log_abs_det_jacobian = log_abs_det_jacobian + log_det
        return out, log_abs_det_jacobian


class TanhTransform(Transform):
    """Transform via the mapping :math:`y = \frac{e^x - e^{-x}} {e^x + e^{-x}}`."""

    @override(Transform)
    def _encode(self, inputs):
        return torch.tanh(inputs)

    @override(Transform)
    def _decode(self, inputs):
        # torch.finfo(torch.float32).tiny
        to_log1 = torch.clamp(1 + inputs, min=1.1754943508222875e-38)
        to_log2 = torch.clamp(1 - inputs, min=1.1754943508222875e-38)
        return (torch.log(to_log1) - torch.log(to_log2)) / 2

    @override(Transform)
    def log_abs_det_jacobian(self, inputs, outputs):
        # pylint:disable=unused-argument
        # Taken from spinningup's implementation of SAC
        return 2 * (math.log(2) - inputs - F.softplus(-2 * inputs))


class SigmoidTransform(Transform):
    @override(Transform)
    def _encode(self, inputs):
        return inputs.sigmoid()

    @override(Transform)
    def _decode(self, inputs):
        to_log = inputs.clamp(min=1.1754943508222875e-38)
        return to_log.log() - (-to_log).log1p()

    @override(Transform)
    def log_abs_det_jacobian(self, inputs, outputs):
        return -F.softplus(-inputs) - F.softplus(inputs)


class AffineTransform(Transform):
    def __init__(self, loc, scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @override(Transform)
    def _encode(self, inputs):
        return inputs * self.scale + self.loc

    @override(Transform)
    def _decode(self, inputs):
        return (inputs - self.loc) / self.scale

    @override(Transform)
    def log_abs_det_jacobian(self, inputs, outputs):
        _, scale = torch.broadcast_tensors(inputs, self.scale)
        return scale.abs().log()


class TanhSquashTransform(ComposeTransform):
    """Squashes samples to the desired range."""

    def __init__(self, low, high, *args, **kwargs):
        squash = TanhTransform(*args, **kwargs)
        shift = AffineTransform(
            loc=(high + low) / 2, scale=(high - low) / 2, *args, **kwargs
        )
        super().__init__([squash, shift])
