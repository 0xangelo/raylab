# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.distributions import SquashedDistribution


class _DistributionBase(nn.Module):
    """Abstract module for preprocessing inputs related to a PyTorch distribution."""

    # pylint: disable=abstract-method

    def __init__(self, dist_cls, low=None, high=None):
        super().__init__()
        self.dist_cls = dist_cls
        assert (low is None) == (
            high is None
        ), "Either both low and high are None or defined"
        if low is None:
            self._squashed = False
        else:
            self._squashed = True
            self.register_buffer("low", low)
            self.register_buffer("high", high)

    def compute_dist(self, inputs):
        """Convenience function to compute the appropriate PyTorch distribution."""
        dist = self.dist_cls(**inputs)
        if self._squashed:
            dist = SquashedDistribution(dist, self.low, self.high)
        return dist


class DistRSample(_DistributionBase):
    """Module producing samples given a dict of distribution parameters."""

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        dist = self.compute_dist(inputs)
        sample = dist.rsample()
        return sample, dist.log_prob(sample)


class DistMean(_DistributionBase):
    """Module producing a distribution's mean given a dict of its parameters."""

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        dist = self.compute_dist(inputs)
        mean = dist.mean
        return mean, dist.log_prob(mean)


class DistLogProb(_DistributionBase):
    """Module producing samples given a dict of distribution parameters."""

    @override(nn.Module)
    def forward(self, inputs, samples):  # pylint: disable=arguments-differ
        dist = self.compute_dist(inputs)
        return dist.log_prob(samples)


class DistReproduce(_DistributionBase):
    """Reproduce a reparametrized sample by inferring the exogenous noise."""

    @override(nn.Module)
    def forward(self, inputs, samples):  # pylint: disable=arguments-differ
        dist = self.compute_dist(inputs)
        return dist.reproduce(samples)
