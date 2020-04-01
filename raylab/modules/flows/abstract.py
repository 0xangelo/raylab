"""Base classes for Normalizing Flows."""
from typing import Dict

import torch
from ray.rllib.utils.annotations import override

from ..distributions import Transform, ConditionalTransform
from ..distributions.utils import _sum_rightmost


class NormalizingFlow(Transform):
    """A diffeomorphism.

    Flows are specialized `Transform`s with tractable Jacobians. They can be used
    in most situations where a `Transform` would be (e.g., with `ComposeTransform`).
    All flows map samples from a latent space to another (f(z) -> x)
    Use the `reverse` flag to invert the flow (f^{-1}(x) -> z).
    """

    @override(Transform)
    def forward(self, inputs, reverse: bool = False):
        if reverse:
            out, log_abs_det_jacobian = self._decode(inputs)
        else:
            out, log_abs_det_jacobian = self._encode(inputs)
        return out, _sum_rightmost(log_abs_det_jacobian, self.event_dim)

    def _encode(self, inputs):
        """
        Apply the forward transformation to the data.

        Maps latent variables to datapoints, returning the transformed variable and the
        log of the absolute Jacobian determinant.
        """
        return None, None

    def _decode(self, inputs):
        """
        Apply the inverse transformation to the data.

        Maps data points to latent variables, returning the transformed variable and the
        log of the absolute Jacobian determinant.
        """
        return None, None


class ConditionalNormalizingFlow(ConditionalTransform):
    """A Normalizing Flow conditioned on some external variable."""

    @override(ConditionalTransform)
    def forward(self, inputs, cond: Dict[str, torch.Tensor], reverse: bool = False):
        if reverse:
            out, log_abs_det_jacobian = self._decode(inputs, cond)
        else:
            out, log_abs_det_jacobian = self._encode(inputs, cond)
        return out, _sum_rightmost(log_abs_det_jacobian, self.event_dim)

    @override(ConditionalTransform)
    def _encode(self, inputs, cond: Dict[str, torch.Tensor]):
        return None, None

    @override(ConditionalTransform)
    def _decode(self, inputs, cond: Dict[str, torch.Tensor]):
        return None, None
