# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.distributions import DiagMultivariateNormal, SquashedMultivariateNormal


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
            self.register_buffer("action_low", action_low)
            self.register_buffer("action_high", action_high)

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
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
        sample = dist.mean if self.mean_only else dist.rsample()
        return sample, dist.log_prob(sample)
