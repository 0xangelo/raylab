"""Neural network modules for SAC."""
import torch.nn as nn


class ActionValueFunction(nn.Module):
    """Neural network module emulating a Q value function."""

    def __init__(self, logits_module, value_module):
        super().__init__()
        self.logits_module = logits_module
        self.value_module = value_module

    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs, actions)
        return self.value_module(logits)
