# pylint: disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import initialize_
from .leaf_parameter import LeafParameter


class DiagMultivariateNormalParams(nn.Module):
    """Neural network module mapping inputs to DiagMultivariateNormal parameters.

    This module is initialized to be close to a standard Normal distribution.
    """

    __constants__ = {"LOG_STD_MAX", "LOG_STD_MIN"}
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self, in_features, event_dim, input_dependent_scale=True):
        super().__init__()
        self.loc_module = nn.Linear(in_features, event_dim)
        if input_dependent_scale:
            self.log_scale_module = nn.Linear(in_features, event_dim)
        else:
            self.log_scale_module = LeafParameter(event_dim)
        self.apply(initialize_("orthogonal", gain=0.01))

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        loc = self.loc_module(inputs)
        log_scale = self.log_scale_module(inputs)
        scale_diag = torch.clamp(log_scale, self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return {"loc": loc, "scale_diag": scale_diag}
