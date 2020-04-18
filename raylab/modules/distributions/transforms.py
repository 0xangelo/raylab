"""Distribution transforms as PyTorch modules compatible with TorchScript."""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils.annotations import override

from .utils import _sum_rightmost


class Transform(nn.Module):
    """A diffeomorphism.

    Transforms are differentiable bijections with tractable Jacobians.
    All transforms map samples from a latent space to another (f(z) -> x)
    Use the `reverse` flag to invert the transformation (f^{-1}(x) -> z).
    """

    params: Dict[str, torch.Tensor]

    def __init__(self, *, cond_transform=None, params=None, event_dim=0):
        super().__init__()
        self.event_dim = (
            event_dim if cond_transform is None else cond_transform.event_dim
        )
        self.cond_transform = cond_transform
        self.params = params or {}
        for name, param in self.params.items():
            if isinstance(param, nn.Parameter):
                self.register_parameter(name, param)
            else:
                self.register_buffer(name, param)

    @override(nn.Module)
    def forward(self, inputs, reverse: bool = False):  # pylint:disable=arguments-differ
        return self.decode(inputs) if reverse else self.encode(inputs)

    def encode(self, inputs):
        """
        Computes the transform `z => x` and the log det jacobian `log |dz/dx|`
        """

        return self.cond_transform.encode(inputs, self.params)

    def decode(self, inputs):
        """
        Inverts the transform `x => z` and the log det jacobian `log |dx/dz|`,
        or `- log |dz/dx|`.
        """

        return self.cond_transform.decode(inputs, self.params)


class ConditionalTransform(nn.Module):
    """A Transform conditioned on some external variable(s)."""

    def __init__(self, *, transform=None, event_dim=0):
        super().__init__()
        self.event_dim = event_dim if transform is None else transform.event_dim
        self.transform = transform

    @override(nn.Module)
    def forward(self, inputs, params: Dict[str, torch.Tensor], reverse: bool = False):
        # pylint:disable=arguments-differ
        return self.decode(inputs, params) if reverse else self.encode(inputs, params)

    def encode(self, inputs, params: Dict[str, torch.Tensor]):
        """
        Computes the transform `(z, y) => x`.
        """
        # pylint:disable=unused-argument
        return self.transform.encode(inputs)

    def decode(self, inputs, params: Dict[str, torch.Tensor]):
        """
        Inverts the transform `(x, y) => z`.
        """
        # pylint:disable=unused-argument
        return self.transform.decode(inputs)


class InverseTransform(ConditionalTransform):
    """Invert the transform, effectively swapping the encoding/decoding directions."""

    def __init__(self, transform):
        super().__init__(event_dim=transform.event_dim)
        self.transform = (
            ConditionalTransform(transform=transform)
            if isinstance(transform, Transform)
            else transform
        )

    @override(ConditionalTransform)
    def encode(self, inputs, params: Dict[str, torch.Tensor]):

        return self.transform.decode(inputs, params)

    @override(ConditionalTransform)
    def decode(self, inputs, params: Dict[str, torch.Tensor]):

        return self.transform.encode(inputs, params)


class CompositeTransform(ConditionalTransform):
    # pylint:disable=missing-docstring

    def __init__(self, transforms, event_dim=None):
        event_dim = event_dim or max(t.event_dim for t in transforms)
        super().__init__(event_dim=event_dim)
        assert self.event_dim >= max(t.event_dim for t in transforms), (
            "CompositeTransform cannot have an event_dim smaller than any "
            "of its components'"
        )
        transforms = self.unpack(transforms)
        self.transforms = nn.ModuleList(transforms)
        self.inv_transforms = nn.ModuleList(transforms[::-1])

    @staticmethod
    def unpack(transforms):
        """Recursively unfold CompositeTransforms in a list."""
        result = []
        for trans in transforms:
            if isinstance(trans, CompositeTransform):
                result.extend(trans.unpack(trans.transforms))
            elif isinstance(trans, Transform):
                result += [ConditionalTransform(transform=trans)]
            else:
                result += [trans]
        return result

    @override(ConditionalTransform)
    def encode(self, inputs, params: Dict[str, torch.Tensor]):
        out = inputs
        log_abs_det_jacobian = 0.0
        for transform in self.transforms:
            out, log_det = transform(out, params, reverse=False)
            log_abs_det_jacobian += _sum_rightmost(
                log_det, self.event_dim - transform.event_dim
            )
        return out, log_abs_det_jacobian

    @override(ConditionalTransform)
    def decode(self, inputs, params: Dict[str, torch.Tensor]):
        out = inputs
        log_abs_det_jacobian = 0.0
        for transform in self.inv_transforms:
            out, log_det = transform(out, params, reverse=True)
            log_abs_det_jacobian += _sum_rightmost(
                log_det, self.event_dim - transform.event_dim
            )
        return out, log_abs_det_jacobian


class TanhTransform(Transform):
    """Transform via the mapping :math:`y = \frac{e^x - e^{-x}} {e^x + e^{-x}}`."""

    # pylint:disable=arguments-out-of-order

    @override(Transform)
    def encode(self, inputs):
        outputs = torch.tanh(inputs)
        return outputs, self.log_abs_det_jacobian(inputs, outputs)

    @override(Transform)
    def decode(self, inputs):
        # torch.finfo(torch.float32).tiny
        to_log1 = torch.clamp(1 + inputs, min=1.1754943508222875e-38)
        to_log2 = torch.clamp(1 - inputs, min=1.1754943508222875e-38)
        outputs = (torch.log(to_log1) - torch.log(to_log2)) / 2
        return outputs, -self.log_abs_det_jacobian(outputs, inputs)

    def log_abs_det_jacobian(self, inputs, outputs):
        # pylint:disable=unused-argument,missing-docstring
        # Taken from spinningup's implementation of SAC
        return _sum_rightmost(
            2 * (math.log(2) - inputs - F.softplus(-2 * inputs)), self.event_dim
        )


class SigmoidTransform(Transform):
    # pylint:disable=missing-docstring,arguments-out-of-order

    @override(Transform)
    def encode(self, inputs):
        outputs = inputs.sigmoid()
        return outputs, self.log_abs_det_jacobian(inputs, outputs)

    @override(Transform)
    def decode(self, inputs):
        to_log = inputs.clamp(min=1.1754943508222875e-38)
        outputs = to_log.log() - (-to_log).log1p()
        return outputs, -self.log_abs_det_jacobian(outputs, inputs)

    def log_abs_det_jacobian(self, inputs, outputs):
        # pylint:disable=unused-argument,missing-docstring
        return _sum_rightmost(-F.softplus(-inputs) - F.softplus(inputs), self.event_dim)


class AffineTransform(Transform):
    # pylint:disable=missing-docstring,arguments-out-of-order

    def __init__(self, loc, scale, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @override(Transform)
    def encode(self, inputs):
        outputs = inputs * self.scale + self.loc
        return outputs, self.log_abs_det_jacobian(inputs, outputs)

    @override(Transform)
    def decode(self, inputs):
        outputs = (inputs - self.loc) / self.scale
        return outputs, -self.log_abs_det_jacobian(outputs, inputs)

    def log_abs_det_jacobian(self, inputs, outputs):
        # pylint:disable=unused-argument,missing-docstring
        _, scale = torch.broadcast_tensors(inputs, self.scale)
        return _sum_rightmost(scale.abs().log(), self.event_dim)


class TanhSquashTransform(Transform):
    """Squashes samples to the desired range using Tanh."""

    def __init__(self, low, high, event_dim=0):
        divide = AffineTransform(loc=torch.zeros_like(low), scale=2 / (high - low))
        squash = TanhTransform()
        shift = AffineTransform(loc=(high + low) / 2, scale=(high - low) / 2)
        compose = CompositeTransform([divide, squash, shift], event_dim=event_dim)
        super().__init__(cond_transform=compose)


class SigmoidSquashTransform(Transform):
    """Squashes samples to the desired range using Sigmoid."""

    def __init__(self, low, high, event_dim=0):
        divide = AffineTransform(loc=torch.zeros_like(low), scale=1 / (high - low))
        squash = SigmoidTransform()
        shift = AffineTransform(loc=low, scale=(high - low))
        compose = CompositeTransform([divide, squash, shift], event_dim=event_dim)
        super().__init__(cond_transform=compose)
