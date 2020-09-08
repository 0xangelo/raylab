"""Modules mapping inputs to distribution parameters."""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.utils import override

from raylab.torch.nn.init import initialize_

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


class PolicyNormalParams(nn.Module):
    # pylint:disable=line-too-long
    """Produce Normal parameters for a stochastic policy.

    Max and min log scale clipping borrowed from `spinningup`_ and `pfrl`_.
    Initialization copied from `pfrl`_.

    .. _`spinningup`: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
    .. _`prfrl`: https://github.com/pfnet/pfrl/blob/master/examples/mujoco/reproduction/soft_actor_critic/train_soft_actor_critic.py
    """
    # pylint:enable=line-too-long
    def __init__(
        self,
        in_features: int,
        event_size: int,
        input_dependent_scale: bool = True,
    ):
        super().__init__()
        self.loc_module = nn.Linear(in_features, event_size)
        if input_dependent_scale:
            self.log_scale_module = nn.Linear(in_features, event_size)
        else:
            self.log_scale_module = LeafParameter(event_size)

        self.apply(initialize_("xavier_uniform", gain=1.0))

    @override(nn.Module)
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # pylint:disable=arguments-differ
        loc = self.loc_module(inputs)

        log_scale = self.log_scale_module(inputs)
        log_scale = torch.clamp(log_scale, min=-20, max=2)
        scale = log_scale.exp()

        return {"loc": loc, "scale": scale}


class NormalParams(nn.Module):
    # pylint:disable=line-too-long
    """Produce Normal parameters.

    Initialized to be close to a standard Normal distribution.

    Utilizes bounded log_stddev as described in the 'Well behaved probabilistic
    networks' appendix of `PETS`_.

    .. _`PETS`: https://papers.nips.cc/paper/7725-deep-reinforcement-learning-in-a-handful-of-trials-using-probabilistic-dynamics-models
    """
    # pylint:enable=line-too-long

    def __init__(
        self,
        in_features: int,
        event_size: int,
        input_dependent_scale: bool = True,
        bound_parameters: bool = True,
    ):
        super().__init__()
        self.loc_module = nn.Linear(in_features, event_size)
        if input_dependent_scale:
            self.log_scale_module = nn.Linear(in_features, event_size)
        else:
            self.log_scale_module = LeafParameter(event_size)

        max_logvar = torch.ones(event_size) / 2.0
        min_logvar = -torch.ones(event_size) * 10.0
        if bound_parameters:
            self.max_logvar = nn.Parameter(max_logvar)
            self.min_logvar = nn.Parameter(min_logvar)
        else:
            self.register_buffer("max_logvar", max_logvar)
            self.register_buffer("min_logvar", min_logvar)

        self.apply(initialize_("orthogonal", gain=0.01))

    @override(nn.Module)
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # pylint:disable=arguments-differ
        loc = self.loc_module(inputs)

        log_scale = self.log_scale_module(inputs)
        max_logvar = self.max_logvar.expand_as(log_scale)
        min_logvar = self.min_logvar.expand_as(log_scale)
        log_scale = max_logvar - F.softplus(max_logvar - log_scale)
        log_scale = min_logvar + F.softplus(log_scale - min_logvar)
        scale = log_scale.exp()

        return {
            "loc": loc,
            "scale": scale,
            "max_logvar": max_logvar,
            "min_logvar": min_logvar,
        }


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
