# pylint:disable=missing-docstring
import torch.nn as nn
from ray.rllib.utils import override


class Lambda(nn.Module):
    """Neural network module that stores and applies a function on inputs."""

    def __init__(self, func):
        super().__init__()
        self.func = func

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.func(inputs)
