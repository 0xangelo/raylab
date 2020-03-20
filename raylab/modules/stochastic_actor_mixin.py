"""Support for modules with stochastic policies."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.distributions import DiagMultivariateNormal
from .basic import (
    FullyConnected,
    DiagMultivariateNormalParams,
    DistMean,
    DistRSample,
    DistLogProb,
    DistReproduce,
)


class StochasticActorMixin:
    """Adds constructor for modules with stochastic policies."""

    # pylint:disable=too-few-public-methods

    @staticmethod
    def _make_actor(obs_space, action_space, config):
        actor = nn.ModuleDict()
        actor_config = config["actor"]

        actor.params = StochasticPolicyParams(obs_space, action_space, actor_config)

        dist_kwargs = dict(
            dist_cls=DiagMultivariateNormal,
            detach_logp=False,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )
        dist_samp = (
            DistMean(**dist_kwargs)
            if config["mean_action_only"]
            else DistRSample(**dist_kwargs)
        )
        dist_logp = DistLogProb(
            dist_cls=DiagMultivariateNormal,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )
        dist_repr = DistReproduce(
            dist_cls=DiagMultivariateNormal,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )
        if config["torch_script"]:
            params_ = {
                "loc": torch.zeros(1, *action_space.shape),
                "scale_diag": torch.ones(1, *action_space.shape),
            }
            actions_ = torch.randn(1, *action_space.shape)
            dist_samp = dist_samp.traced(params_)
            dist_logp = dist_logp.traced(params_, actions_)
            dist_repr = dist_repr.traced(params_, actions_)
        actor.rsample = nn.Sequential(actor.params, dist_samp)
        actor.logp = PolicyLogProb(actor.params, dist_logp)
        actor.reproduce = PolicyReproduce(actor.params, dist_repr)
        return {"actor": actor}


class MaximumEntropyMixin:
    """Adds entropy coefficient parameter to module."""

    # pylint:disable=too-few-public-methods

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self.log_alpha = nn.Parameter(torch.zeros([]))


class StochasticPolicyParams(nn.Module):
    """Represents a stochastic policy as a sequence of modules."""

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.logits = FullyConnected(
            in_features=obs_space.shape[0],
            units=config["units"],
            activation=config["activation"],
            **config["initializer_options"]
        )
        self.params = DiagMultivariateNormalParams(
            self.logits.out_features,
            action_space.shape[0],
            input_dependent_scale=config["input_dependent_scale"],
        )
        self.sequential = nn.Sequential(self.logits, self.params)

    @override(nn.Module)
    def forward(self, obs):  # pylint:disable=arguments-differ
        return self.sequential(obs)


class PolicyLogProb(nn.Module):
    """Computes the log-likelihood of actions."""

    def __init__(self, params_module, logp_module):
        super().__init__()
        self.params_module = params_module
        self.logp_module = logp_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint:disable=arguments-differ
        return self.logp_module(self.params_module(obs), actions)


class PolicyReproduce(nn.Module):
    """Reproduces observed actions."""

    def __init__(self, params_module, resample_module):
        super().__init__()
        self.params_module = params_module
        self.resample_module = resample_module

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint:disable=arguments-differ
        return self.resample_module(self.params_module(obs), actions)
