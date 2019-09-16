# pylint: disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class ActionModule(nn.Module):
    """Neural network module mapping inputs to actions in specified range."""

    __constants__ = {"in_features", "out_features"}

    def __init__(self, in_features, action_low, action_high):
        super().__init__()
        self.in_features = in_features
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_range", action_high - action_low)
        self.out_features = self.action_low.numel()
        self.linear_module = nn.Linear(self.in_features, self.out_features)

    @override(nn.Module)
    def forward(self, logits):  # pylint: disable=arguments-differ
        unscaled_actions = self.linear_module(logits)
        squashed_actions = torch.sigmoid(unscaled_actions / self.action_range)
        scaled_actions = squashed_actions * self.action_range + self.action_low
        return scaled_actions
