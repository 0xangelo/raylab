"""Actor-Critic architecture with deterministic actor."""
import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from raylab.distributions import DiagMultivariateNormal
from .basic import FullyConnected, DiagMultivariateNormalParams, DistMean, DistRSample
from .deterministic_actor_critic import DeterministicActorCritic


class StochasticActorCritic(DeterministicActorCritic):
    """Module containing stochastic policy and action value functions."""

    # pylint:disable=abstract-method

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self.log_alpha = nn.Parameter(torch.zeros([]))

    @override(DeterministicActorCritic)
    def _make_policy(self, obs_space, action_space, config):
        policy_module = StochasticPolicy(obs_space, action_space, config["policy"])
        if self._script:
            policy_module = policy_module.as_script_module()

        dist_kwargs = dict(
            dist_cls=DiagMultivariateNormal,
            detach_logp=False,
            low=torch.as_tensor(action_space.low),
            high=torch.as_tensor(action_space.high),
        )
        sampler_module = nn.Sequential(
            policy_module,
            DistMean(**dist_kwargs)
            if config["mean_action_only"]
            else DistRSample(**dist_kwargs),
        )
        return {"policy": policy_module, "sampler": sampler_module}


class StochasticPolicy(nn.Module):
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

    def as_script_module(self):
        """Return self as a ScriptModule."""
        self.logits = self.logits.as_script_module()
        self.params = self.params.as_script_module()
        self.sequential = nn.Sequential(self.logits, self.params)
        return torch.jit.script(self)
