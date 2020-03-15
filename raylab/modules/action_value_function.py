# pylint: disable=missing-docstring
# pylint: enable=missing-docstring
import torch.nn as nn
from ray.rllib.utils.annotations import override

from .basic import StateActionEncoder
from .value_function import ValueFunction


class ActionValueFunction(nn.Module):
    """Neural network module emulating a Q value function."""

    def __init__(self, logits_module, value_module):
        super().__init__()
        self.logits_module = logits_module
        self.value_module = value_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs, actions)
        return self.value_module(logits)

    @classmethod
    def from_scratch(cls, *logits_args, **logits_kwargs):
        """Create an action value function with new logits and value modules."""
        logits_module = StateActionEncoder(*logits_args, **logits_kwargs)
        value_module = ValueFunction(logits_module.out_features)
        return cls(logits_module, value_module)
