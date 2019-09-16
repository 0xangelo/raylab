# pylint: disable=missing-docstring
import torch.nn as nn
from ray.rllib.utils.annotations import override


class ValueModule(nn.Module):
    """Neural network module mapping inputs to value function outputs."""

    __constants__ = {"in_features", "out_features"}

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = 1
        self.linear_module = nn.Linear(self.in_features, self.out_features)

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        return self.linear_module(logits)
