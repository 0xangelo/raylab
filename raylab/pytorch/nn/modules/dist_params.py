"""Modules mapping inputs to distribution parameters."""
from typing import Dict

import torch
import torch.nn as nn
from ray.rllib.utils import override

from raylab.pytorch.nn.init import initialize_

from .leaf_parameter import LeafParameter


class CategoricalParams(nn.Module):
    """Produce Categorical parameters.

    Initialized to be close to a discrete uniform distribution.
    """

    def __init__(self, in_features: int, n_categories: int):
        super().__init__()
        self.logits_module = nn.Linear(in_features, n_categories)
        self.apply(initialize_("orthogonal", gain=0.01))

    @override(nn.Module)
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # pylint:disable=arguments-differ
        logits = self.logits_module(inputs)
        return {"logits": logits - logits.logsumexp(dim=-1, keepdim=True)}


class NormalParams(nn.Module):
    """Produce Normal parameters.

    Initialized to be close to a standard Normal distribution.
    """

    __constants__ = {"LOG_STD_MAX", "LOG_STD_MIN"}
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(
        self, in_features: int, event_size: int, input_dependent_scale: bool = True
    ):
        super().__init__()
        self.loc_module = nn.Linear(in_features, event_size)
        if input_dependent_scale:
            self.log_scale_module = nn.Linear(in_features, event_size)
        else:
            self.log_scale_module = LeafParameter(event_size)
        self.apply(initialize_("orthogonal", gain=0.01))

    @override(nn.Module)
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # pylint:disable=arguments-differ
        loc = self.loc_module(inputs)
        log_scale = self.log_scale_module(inputs)
        scale = torch.clamp(log_scale, self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return {"loc": loc, "scale": scale}


class StdNormalParams(nn.Module):
    """Produce Normal parameters with unit variance."""

    def __init__(self, input_dim: int, event_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.event_shape = (event_size,)

    @override(nn.Module)
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # pylint:disable=arguments-differ
        shape = inputs.shape[: -self.input_dim] + self.event_shape
        return {"loc": torch.zeros(shape), "scale": torch.ones(shape)}
