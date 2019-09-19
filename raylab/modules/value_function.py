# pylint: disable=missing-docstring
import torch.nn as nn


class ValueFunction(nn.Linear):
    """Neural network module mapping inputs to value function outputs."""

    def __init__(self, in_features):
        super().__init__(in_features, 1)
