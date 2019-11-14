"""Distribution operations that don't detach the samples from the graph."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.modules import DistRSample as DistRSample_, DistMean as DistMean_


class DistRSample(DistRSample_):
    """Module producing samples given a dict of distribution parameters."""

    @override(nn.Module)
    def forward(
        self, inputs, sample_shape=torch.Size([])
    ):  # pylint: disable=arguments-differ
        dist = self.compute_dist(inputs)
        sample = dist.rsample(sample_shape=sample_shape)
        return sample, dist.log_prob(sample)


class DistMean(DistMean_):
    """Module producing a distribution's mean given a dict of its parameters."""

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        dist = self.compute_dist(inputs)
        mean = dist.mean
        return mean, dist.log_prob(mean)
