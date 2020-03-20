# pylint: disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from .fully_connected import FullyConnected


class StateActionEncoder(nn.Module):
    """Neural network module which concatenates action after the first layer."""

    __constants__ = {"in_features", "out_features"}

    def __init__(self, obs_dim, action_dim, delay_action=True, units=(), **fc_kwargs):
        super().__init__()
        self.in_features = obs_dim + action_dim
        if units:
            if delay_action is True:
                self.obs_module = FullyConnected(obs_dim, units=units[:1], **fc_kwargs)
                input_dim = units[0] + action_dim
                units = units[1:]
            else:
                self.obs_module = nn.Identity()
                input_dim = obs_dim + action_dim
            self.sequential_module = FullyConnected(input_dim, units=units, **fc_kwargs)
            self.out_features = self.sequential_module.out_features
        else:
            self.obs_module = nn.Identity()
            self.sequential_module = nn.Identity()
            self.out_features = obs_dim + action_dim

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        output = self.obs_module(obs)
        output = torch.cat([output, actions], dim=-1)
        output = self.sequential_module(output)
        return output
