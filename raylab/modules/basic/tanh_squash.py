# pylint: disable=missing-docstring
import torch.nn as nn
from ray.rllib.utils.annotations import override


class TanhSquash(nn.Module):
    """Neural network module squashing vectors to specified range using Tanh."""

    def __init__(self, low, high):
        super().__init__()
        self.register_buffer("loc", (high + low) / 2)
        self.register_buffer("scale", (high - low) / 2)

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self.loc + inputs.tanh() * self.scale
