# pylint:disable=missing-docstring
import torch.nn as nn

from .linear import NormalizedLinear
from .tanh_squash import TanhSquash


class ActionOutput(nn.Sequential):
    """Neural network module mapping inputs to actions in specified range."""

    __constants__ = {"in_features", "out_features"}

    def __init__(self, in_features, action_low, action_high, *, beta):
        self.in_features = in_features
        self.out_features = action_low.shape[-1]
        super().__init__(
            NormalizedLinear(self.in_features, self.out_features, beta=beta),
            TanhSquash(action_low, action_high),
        )
