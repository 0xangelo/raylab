# pylint:disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils import override


class LeafParameter(nn.Module):
    """Holds a single paramater vector an expands it to match batch shape of inputs."""

    def __init__(self, in_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(in_features))

    @override(nn.Module)
    def forward(self, inputs):  # pylint:disable=arguments-differ
        return self.bias.expand(inputs.shape[:-1] + (-1,))
