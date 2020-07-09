# pylint:disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils import override


class GaussianNoise(nn.Module):
    """Neural network module adding gaussian i.i.d. noise to inputs."""

    __constants__ = {"scale"}

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return inputs + torch.randn_like(inputs) * self.scale
