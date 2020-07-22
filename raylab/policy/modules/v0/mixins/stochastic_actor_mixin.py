"""Support for modules with stochastic policies."""
from typing import List

import gym.spaces as spaces
import torch
import torch.nn as nn
from ray.rllib.utils import override

import raylab.pytorch.nn as nnx
import raylab.pytorch.nn.distributions as ptd
from raylab.pytorch.nn.init import initialize_
from raylab.utils.dictionaries import deep_merge


def _build_fully_connected(obs_space, config):
    mlp = nnx.FullyConnected(in_features=obs_space.shape[0], **config["encoder"])
    mlp.apply(
        initialize_(
            activation=config["encoder"].get("activation"),
            **config["initializer_options"],
        )
    )
    return mlp


class StochasticActorMixin:
    """Adds constructor for modules with stochastic policies."""

    # pylint:disable=too-few-public-methods
    BASE_CONFIG = {
        "encoder": {"units": (32, 32), "activation": "Tanh"},
        "initializer_options": {"name": "xavier_uniform"},
        "input_dependent_scale": False,
    }

    @staticmethod
    def _make_actor(obs_space, action_space, config):
        config = deep_merge(
            StochasticActorMixin.BASE_CONFIG,
            config.get("actor", {}),
            False,
            ["encoder"],
        )

        if isinstance(action_space, spaces.Discrete):
            logits = _build_fully_connected(obs_space, config)
            params = nnx.CategoricalParams(logits.out_features, action_space.n)
            params_module = nn.Sequential(logits, params)
            dist_module = ptd.Categorical()
        elif isinstance(action_space, spaces.Box):
            logits = _build_fully_connected(obs_space, config)
            params = nnx.NormalParams(
                logits.out_features,
                action_space.shape[0],
                input_dependent_scale=config["input_dependent_scale"],
            )
            params_module = nn.Sequential(logits, params)
            dist_module = ptd.TransformedDistribution(
                ptd.Independent(ptd.Normal(), reinterpreted_batch_ndims=1),
                ptd.flows.TanhSquashTransform(
                    low=torch.as_tensor(action_space.low),
                    high=torch.as_tensor(action_space.high),
                    event_dim=1,
                ),
            )
        else:
            raise ValueError(f"Unsopported action space type {type(action_space)}")

        return {"actor": StochasticPolicy(params_module, dist_module)}


class MaximumEntropyMixin:
    """Adds entropy coefficient parameter to module."""

    # pylint:disable=too-few-public-methods
    BASE_CONFIG = {"initial_alpha": 1.0}

    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)
        config = deep_merge(
            MaximumEntropyMixin.BASE_CONFIG, config.get("entropy", {}), False
        )
        self.alpha = Alpha(config["initial_alpha"])


class StochasticPolicy(nn.Module):
    """Represents a stochastic policy as a conditional distribution module."""

    # pylint:disable=abstract-method

    def __init__(self, params_module, dist_module):
        super().__init__()
        self.params = params_module
        self.dist = dist_module

    @override(nn.Module)
    def forward(self, obs):  # pylint:disable=arguments-differ
        return self.params(obs)

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
        return self.dist.log_prob(action, params)

    @torch.jit.export
    def cdf(self, obs, action):
        """Returns the cumulative density/mass function evaluated at `action`."""
        params = self(obs)
        return self.dist.cdf(action, params)

    @torch.jit.export
    def icdf(self, obs, prob):
        """Returns the inverse cumulative density/mass function evaluated at `prob`."""
        params = self(obs)
        return self.dist.icdf(prob, params)

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
        return self.dist.reproduce(action, params)

    @torch.jit.export
    def deterministic(self, obs):
        """
        Generates a deterministic sample or batch of samples if the distribution
        parameters are batched. Returns a (rsample, log_prob) pair.
        """
        params = self(obs)
        return self.dist.deterministic(params)


class Alpha(nn.Module):
    # pylint:disable=missing-class-docstring

    def __init__(self, initial_alpha):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.as_tensor(initial_alpha).log())

    def forward(self):  # pylint:disable=arguments-differ
        return self.log_alpha.exp()
