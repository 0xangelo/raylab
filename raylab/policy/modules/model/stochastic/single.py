"""NN modules for stochastic dynamics estimation."""
from dataclasses import dataclass
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


@dataclass
class MLPModelSpec(StateActionMLP.spec_cls):
    """Specifications for stochastic mlp model network.

    Inherits parameters from `StateActionMLP.spec_cls`.

    Args:
        units: Number of units in each hidden layer
        activation: Nonlinearity following each linear layer
        delay_action: Whether to apply an initial preprocessing layer on the
            observation before concatenating the action to the input.
        standard_scaler: Whether to transform the inputs of the NN using a
            standard scaling procedure (subtract mean and divide by stddev). The
            transformation mean and stddev should be fitted during training and
            used for both training and evaluation.
        fix_logvar_bounds: Whether to use fixed or dynamically adjusted
            bounds for the log-scale outputs of the network.
        input_dependent_scale: Whether to parameterize the Gaussian standard
            deviation as a function of the state and action
    """

    fix_logvar_bounds: bool = True
    input_dependent_scale: bool = True


class MLPModel(StochasticModel):
    """Stochastic model with multilayer perceptron state-action encoder.

    Attributes:
        params: NN module mapping obs-act pairs to obs dist params
        dist: NN module implementing the distribution API
        encoder: NN module used in `params` to map obs-act pairs to vector
            embeddings
    """

    spec_cls = MLPModelSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: MLPModelSpec):
        encoder = StateActionMLP(obs_space, action_space, spec)

        params = nnx.NormalParams(
            encoder.out_features,
            obs_space.shape[0],
            input_dependent_scale=spec.input_dependent_scale,
            bound_parameters=not spec.fix_logvar_bounds,
        )
        if spec.fix_logvar_bounds:
            params.max_logvar.fill_(2)
            params.min_logvar.fill_(-20)
        params = DynamicsParams(encoder, params)

        dist = ptd.Independent(ptd.Normal(), reinterpreted_batch_ndims=1)

        super().__init__(params, dist)
        # Can only assign modules and parameters after calling nn.Module.__init__
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
