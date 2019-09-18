# pylint: disable=missing-docstring
import torch.nn as nn
from ray.rllib.utils.annotations import override


class DiagMultivariateNormalParams(nn.Module):
    """Neural network module mapping inputs DiagMultivariateNormal parameters."""

    def __init__(self, in_features, event_dim):
        super().__init__()
        self.loc_module = nn.Linear(in_features, event_dim)
        self.scale_diag_module = nn.Linear(in_features, event_dim)

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        loc = self.loc_module(inputs)
        scale_diag = self.scale_diag_module(inputs)
        return loc, scale_diag
