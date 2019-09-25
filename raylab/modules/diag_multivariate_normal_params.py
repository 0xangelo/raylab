# pylint: disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override


class ExpandVector(nn.Module):
    """Holds a single paramater vector an expands it to match batch shape of inputs."""

    def __init__(self, in_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(in_features))

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self.bias.expand(inputs.shape[:-1] + (-1,))


class DiagMultivariateNormalParams(nn.Module):
    """Neural network module mapping inputs DiagMultivariateNormal parameters."""

    def __init__(self, in_features, event_dim, input_dependent_scale=True):
        super().__init__()
        self.loc_module = nn.Linear(in_features, event_dim)
        if input_dependent_scale:
            self.pre_scale_module = nn.Linear(in_features, event_dim)
        else:
            self.pre_scale_module = ExpandVector(event_dim)
        self.softplus = nn.Softplus()

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        loc = self.loc_module(inputs)
        unnormalized_scale = self.pre_scale_module(inputs)
        scale_diag = self.softplus(unnormalized_scale)
        return dict(loc=loc, scale_diag=scale_diag)
