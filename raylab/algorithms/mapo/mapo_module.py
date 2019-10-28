"""Neural Network modules for Model-Aware Policy Optimization."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

import raylab.modules as mods
from raylab.distributions import DiagMultivariateNormal


class DynamicsModel(nn.Module):
    """Neural network module mapping state-action pairs to distribution parameters."""

    def __init__(self, logits_module, params_module):
        super().__init__()
        self.logits_module = logits_module
        self.params_module = params_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint: disable=arguments-differ
        logits = self.logits_module(obs, actions)
        params = self.params_module(logits)
        return params

    @classmethod
    def from_scratch(cls, obs_dim, input_dependent_scale=True, **logits_kwargs):
        """Create a dynamics model with new logtis and params modules."""
        logits_module = mods.StateActionEncoder(**logits_kwargs)
        params_module = mods.DiagMultivariateNormalParams(
            logits_module.out_features,
            obs_dim,
            input_dependent_scale=input_dependent_scale,
        )
        return cls(logits_module, params_module)


class DynamicsModelRSample(nn.Module):
    """
    Neural network module producing samples and log probs given state-action pairs.
    """

    def __init__(self, dynamics_module):
        super().__init__()
        self.dynamics_module = dynamics_module
        self.rsample_module = mods.DistRSample(DiagMultivariateNormal)

    @override(nn.Module)
    def forward(
        self, obs, actions, sample_shape=torch.Size([])
    ):  # pylint: disable=arguments-differ
        params = self.dynamics_module(obs, actions)
        sample, logp = self.rsample_module(params, sample_shape=sample_shape)
        return sample, logp
