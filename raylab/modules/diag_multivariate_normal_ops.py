# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.distributions import DiagMultivariateNormal, SquashedMultivariateNormal


class _DiagMultivariateNormalBase(nn.Module):
    """Abstract module for preprocessing inputs related to a DiagMultivariateNormal."""

    # pylint: disable=abstract-method

    __constants__ = {"mean_only", "squashed"}

    def __init__(
        self, mean_only=False, squashed=False, action_low=None, action_high=None
    ):
        super().__init__()
        self.mean_only = mean_only
        self.squashed = squashed
        if self.squashed:
            assert not (
                action_low is None or action_high is None
            ), "Squashing actions requires upper and lower bounds"
            self.register_buffer("action_low", action_low)
            self.register_buffer("action_high", action_high)

    def compute_dist(self, inputs):
        """Convenience function to compute the appropriate PyTorch distribution."""
        if self.squashed:
            dist = SquashedMultivariateNormal(
                loc=inputs["loc"],
                scale_diag=inputs["scale_diag"],
                low=self.action_low,
                high=self.action_high,
            )
        else:
            dist = DiagMultivariateNormal(
                loc=inputs["loc"], scale_diag=inputs["scale_diag"]
            )
        return dist


class DiagMultivariateNormalRSample(_DiagMultivariateNormalBase):
    """Module producing samples given a dict of DiagMultivariateNormal parameters."""

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        dist = self.compute_dist(inputs)
        sample = dist.mean if self.mean_only else dist.rsample()
        return sample, dist.log_prob(sample)


class DiagMultivariateNormalLogProb(_DiagMultivariateNormalBase):
    """Module producing samples given a dict of DiagMultivariateNormal parameters."""

    @override(nn.Module)
    def forward(self, inputs, samples):  # pylint: disable=arguments-differ
        dist = self.compute_dist(inputs)
        return dist.log_prob(samples)


class DiagMultivariateNormalReproduce(_DiagMultivariateNormalBase):
    """Reproduce a reparametrized Normal sample by inferring the exogenous noise."""

    @staticmethod
    def reproduce_value(inputs, samples):
        """Convenience function to reproduce Normal sample given its parameters."""
        loc, scale_diag = inputs["loc"], inputs["scale_diag"]
        with torch.no_grad():
            eps = (samples - loc) / scale_diag
        return loc + scale_diag * eps

    @override(nn.Module)
    def forward(self, inputs, samples):  # pylint: disable=arguments-differ
        if self.squashed:
            dist = self.compute_dist(inputs)
            for transform in reversed(dist.transforms):
                samples = transform.inv(samples)
            _samples = self.reproduce_value(inputs, samples)
            for transform in dist.transforms:
                _samples = transform(_samples)
        else:
            _samples = self.reproduce_value(inputs, samples)
        return _samples
