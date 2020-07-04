"""NN modules for stochastic dynamics estimation."""
from typing import List

import torch
import torch.nn as nn
from gym.spaces import Box

import raylab.pytorch.nn as nnx
import raylab.pytorch.nn.distributions as ptd
from raylab.policy.modules.networks.mlp import StateActionMLP


class StochasticModel(nn.Module):
    """Represents a stochastic model as a conditional distribution module."""

    def __init__(
        self, params_module: nn.Module, dist_module: ptd.ConditionalDistribution
    ):
        super().__init__()
        self.params = params_module
        self.dist = dist_module

    def forward(self, obs, action):  # pylint:disable=arguments-differ
        return self.params(obs, action)

    @torch.jit.export
    def sample(self, obs, action, sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Returns a (sample, log_prob)
        pair.
        """
        params = self(obs, action)
        return self.dist.sample(params, sample_shape)

    @torch.jit.export
    def rsample(self, obs, action, sample_shape: List[int] = ()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Returns a (rsample, log_prob) pair.
        """
        params = self(obs, action)
        return self.dist.rsample(params, sample_shape)

    @torch.jit.export
    def log_prob(self, obs, action, next_obs):
        """
        Returns the log probability density/mass function evaluated at `next_obs`.
        """
        params = self(obs, action)
        return self.dist.log_prob(next_obs, params)

    @torch.jit.export
    def cdf(self, obs, action, next_obs):
        """Returns the cumulative density/mass function evaluated at `next_obs`."""
        params = self(obs, action)
        return self.dist.cdf(next_obs, params)

    @torch.jit.export
    def icdf(self, obs, action, prob):
        """Returns the inverse cumulative density/mass function evaluated at `prob`."""
        params = self(obs, action)
        return self.dist.icdf(prob, params)

    @torch.jit.export
    def entropy(self, obs, action):
        """Returns entropy of distribution."""
        params = self(obs, action)
        return self.dist.entropy(params)

    @torch.jit.export
    def perplexity(self, obs, action):
        """Returns perplexity of distribution."""
        params = self(obs, action)
        return self.dist.perplexity(params)

    @torch.jit.export
    def reproduce(self, obs, action, next_obs):
        """Produce a reparametrized sample with the same value as `next_obs`."""
        params = self(obs, action)
        return self.dist.reproduce(next_obs, params)


class ResidualMixin:
    """Overrides StochasticModel interface to model state transition residuals."""

    # pylint:disable=missing-function-docstring,not-callable

    @torch.jit.export
    def sample(self, obs, action, sample_shape: List[int] = ()):
        params = self(obs, action)
        res, log_prob = self.dist.sample(params, sample_shape)
        return obs + res, log_prob

    @torch.jit.export
    def rsample(self, obs, action, sample_shape: List[int] = ()):
        params = self(obs, action)
        res, log_prob = self.dist.rsample(params, sample_shape)
        return obs + res, log_prob

    @torch.jit.export
    def log_prob(self, obs, action, next_obs):
        params = self(obs, action)
        return self.dist.log_prob(next_obs - obs, params)

    @torch.jit.export
    def cdf(self, obs, action, next_obs):
        params = self(obs, action)
        return self.dist.cdf(next_obs - obs, params)

    @torch.jit.export
    def icdf(self, obs, action, prob):
        params = self(obs, action)
        residual = self.dist.icdf(prob, params)
        return obs + residual

    @torch.jit.export
    def reproduce(self, obs, action, next_obs):
        params = self(obs, action)
        sample_, log_prob_ = self.dist.reproduce(next_obs - obs, params)
        return obs + sample_, log_prob_


class DynamicsParams(nn.Module):
    """Neural network mapping state-action pairs to distribution parameters.

    Args:
        encoder: Module mapping state-action pairs to 1D features
        params: Module mapping 1D features to distribution parameters
    """

    def __init__(self, encoder: nn.Module, params: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.params = params

    def forward(self, obs, actions):  # pylint:disable=arguments-differ
        return self.params(self.encoder(obs, actions))


MLPSpec = StateActionMLP.spec_cls


class MLPModel(StochasticModel):
    """Stochastic model with multilayer perceptron state-action encoder."""

    spec_cls = MLPSpec

    def __init__(
        self,
        obs_space: Box,
        action_space: Box,
        spec: MLPSpec,
        input_dependent_scale: bool,
    ):
        encoder = StateActionMLP(obs_space, action_space, spec)
        params = nnx.NormalParams(
            encoder.out_features,
            obs_space.shape[0],
            input_dependent_scale=input_dependent_scale,
        )
        params = DynamicsParams(encoder, params)
        dist = ptd.Independent(ptd.Normal(), reinterpreted_batch_ndims=1)
        super().__init__(params, dist)
        self.encoder = encoder

    def initialize_parameters(self, initializer_spec: dict):
        """Initialize all encoder parameters.

        Args:
            initializer_spec: Dictionary with mandatory `name` key corresponding
                to the initializer function name in `torch.nn.init` and optional
                keyword arguments.
        """
        self.encoder.initialize_parameters(initializer_spec)


class ResidualMLPModel(ResidualMixin, MLPModel):
    """Residual stochastic multilayer perceptron model."""
