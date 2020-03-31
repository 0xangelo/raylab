"""Support for modules with stochastic policies."""
from typing import List

import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override
import gym.spaces as spaces

from .basic import CategoricalParams, FullyConnected, NormalParams
from .distributions import (
    Categorical,
    Independent,
    Normal,
    TanhSquashTransform,
    TransformedDistribution,
)


class StochasticActorMixin:
    """Adds constructor for modules with stochastic policies."""

    # pylint:disable=too-few-public-methods

    @staticmethod
    def _make_actor(obs_space, action_space, config):
        actor_config = config["actor"]
        return {"actor": StochasticPolicy(obs_space, action_space, actor_config)}


class MaximumEntropyMixin:
    """Adds entropy coefficient parameter to module."""

    # pylint:disable=too-few-public-methods

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        self.log_alpha = nn.Parameter(torch.zeros([]))


class StochasticPolicy(nn.Module):
    """Represents a stochastic policy as a conditional distribution module."""

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.logits = FullyConnected(
            in_features=obs_space.shape[0],
            units=config["units"],
            activation=config["activation"],
            **config["initializer_options"],
        )

        if isinstance(action_space, spaces.Discrete):
            self.params = CategoricalParams(self.logits.out_features, action_space.n)
            self.dist = Categorical()
        elif isinstance(action_space, spaces.Box):
            self.params = NormalParams(
                self.logits.out_features,
                action_space.shape[0],
                input_dependent_scale=config["input_dependent_scale"],
            )
            self.dist = TransformedDistribution(
                Independent(Normal(), reinterpreted_batch_ndims=1),
                TanhSquashTransform(
                    low=torch.as_tensor(action_space.low),
                    high=torch.as_tensor(action_space.high),
                    event_dim=1,
                ),
            )
        else:
            raise ValueError(f"Unsopported action space type {type(action_space)}")
        self.sequential = nn.Sequential(self.logits, self.params)

    @override(nn.Module)
    def forward(self, obs):  # pylint:disable=arguments-differ
        return self.sequential(obs)

    @torch.jit.export
    def sample(self, obs, sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Returns a (sample, log_prob)
        pair.
        """
        params = self(obs)
        return self.dist.sample(params, sample_shape)

    @torch.jit.export
    def rsample(self, obs, sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Returns a (rsample, log_prob) pair.
        """
        params = self(obs)
        return self.dist.rsample(params, sample_shape)

    @torch.jit.export
    def log_prob(self, obs, action):
        """
        Returns the log of the probability density/mass function evaluated at `action`.
        """
        params = self(obs)
        return self.dist.log_prob(params, action)

    @torch.jit.export
    def cdf(self, obs, action):
        """Returns the cumulative density/mass function evaluated at `action`."""
        params = self(obs)
        return self.dist.cdf(params, action)

    @torch.jit.export
    def icdf(self, obs, prob):
        """Returns the inverse cumulative density/mass function evaluated at `prob`."""
        params = self(obs)
        return self.dist.icdf(params, prob)

    @torch.jit.export
    def entropy(self, obs):
        """Returns entropy of distribution."""
        params = self(obs)
        return self.dist.entropy(params)

    @torch.jit.export
    def perplexity(self, obs):
        """Returns perplexity of distribution."""
        params = self(obs)
        return self.dist.perplexity(params)

    @torch.jit.export
    def reproduce(self, obs, action):
        """Produce a reparametrized sample with the same value as `action`."""
        params = self(obs)
        return self.dist.reproduce(params, action)
