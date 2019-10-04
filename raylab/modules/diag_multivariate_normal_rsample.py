# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import torch.nn as nn
import torch.distributions as dists
from ray.rllib.utils.annotations import override

from raylab.distributions import DiagMultivariateNormal
from raylab.distributions.transforms import TanhTransform


class DiagMultivariateNormalRSample(nn.Module):
    """Module producing samples given a dict of DiagMultivariateNormal parameters."""

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
            self.register_buffer("action_loc", (action_high + action_low) / 2)
            self.register_buffer("action_scale", (action_high - action_low) / 2)

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        dist = DiagMultivariateNormal(
            loc=inputs["loc"], scale_diag=inputs["scale_diag"]
        )
        if self.squashed:
            squash = TanhTransform(cache_size=1)
            shift = dists.AffineTransform(
                loc=self.action_loc, scale=self.action_scale, cache_size=1, event_dim=1
            )
            dist = dists.TransformedDistribution(dist, [squash, shift])
        sample = dist.mean if self.mean_only else dist.rsample()
        return sample, dist.log_prob(sample)
