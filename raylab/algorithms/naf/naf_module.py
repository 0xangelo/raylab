"""Normalized Advantage Function nn.Module."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class NAFModule(nn.Module):
    """Neural network module that implements the forward pass of NAF."""

    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        activation = get_activation(config["activation"])
        layers = []
        sizes = config["layers"]
        if sizes:
            self.obs_layer = nn.Sequential(
                nn.Linear(self.obs_dim, sizes[0]), activation()
            )
            sizes[0] += self.action_dim
            for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(activation())
            layers.append(nn.Linear(sizes[-1], 1))
        else:
            self.obs_layer = nn.Identity()
            layers.append(nn.Linear(self.obs_dim + self.action_dim, 1))
        self.layers = nn.ModuleList(layers)

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        output = self.obs_layer(obs)
        output = torch.cat([output, actions], dim=-1)
        for layer in self.layers:
            output = layer(output)
        return output


def get_activation(activation):
    if isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU
        if activation == "elu":
            return nn.ELU
        if activation == "tanh":
            return nn.Tanh
        raise NotImplementedError("Unsupported activation name '{}'".format(activation))
    raise ValueError(
        "'activation' must be a string type, got '{}'".format(type(activation))
    )
