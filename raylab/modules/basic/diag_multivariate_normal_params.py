# pylint: disable=missing-docstring
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.utils.pytorch import initialize_


class ExpandVector(nn.Module):
    """Holds a single paramater vector an expands it to match batch shape of inputs."""

    def __init__(self, in_features):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(in_features))

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        return self.bias.expand(inputs.shape[:-1] + (-1,))


class DiagMultivariateNormalParams(nn.Module):
    """Neural network module mapping inputs to DiagMultivariateNormal parameters.

    This module is initialized to be closed to a standard Normal distribution.
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
            self.log_scale_module = ExpandVector(event_dim)
        self.apply(initialize_("orthogonal", gain=0.01))

    @override(nn.Module)
    def forward(self, inputs):  # pylint: disable=arguments-differ
        loc = self.loc_module(inputs)
        log_scale = self.log_scale_module(inputs)
        scale_diag = torch.clamp(log_scale, self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return {"loc": loc, "scale_diag": scale_diag}

    @classmethod
    def as_script_module(cls, *args, **kwargs):
        return torch.jit.script(cls(*args, **kwargs))
