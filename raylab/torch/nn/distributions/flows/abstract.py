"""Distribution transforms as PyTorch modules compatible with TorchScript."""
from typing import Dict

import torch
import torch.nn as nn
from ray.rllib.utils import override

from .utils import sum_rightmost


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
            log_abs_det_jacobian += sum_rightmost(
                log_det, self.event_dim - transform.event_dim
            )
        return out, log_abs_det_jacobian

    @override(ConditionalTransform)
    def decode(self, inputs, params: Dict[str, torch.Tensor]):
        out = inputs
        log_abs_det_jacobian = 0.0
        for transform in self.inv_transforms:
            out, log_det = transform(out, params, reverse=True)
            log_abs_det_jacobian += sum_rightmost(
                log_det, self.event_dim - transform.event_dim
            )
        return out, log_abs_det_jacobian
