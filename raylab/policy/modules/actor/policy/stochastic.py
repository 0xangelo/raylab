"""Parameterized stochastic policies."""
from typing import Callable
from typing import List

import torch
import torch.nn as nn
from gym.spaces import Box
from gym.spaces import Discrete

import raylab.pytorch.nn as nnx
import raylab.pytorch.nn.distributions as ptd
from raylab.policy.modules.networks.mlp import StateMLP


class StochasticPolicy(nn.Module):
    """Represents a stochastic policy as a conditional distribution module."""

    # pylint:disable=abstract-method

    def __init__(self, params_module, dist_module):
        super().__init__()
        self.params = params_module
        self.dist = dist_module

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


class MLPStochasticPolicy(StochasticPolicy):
    """Stochastic policy with multilayer perceptron state encoder.

    Args:
        obs_space: Observation space
        spec: Specifications for the encoder
        params_fn: Callable that builds a module for computing distribution
            parameters given the number of state features
        dist: Conditional distribution module

    Attributes:
        encoder: Multilayer perceptron state encoder
        spec: MLP spec instance
    """

    spec_cls = StateMLP.spec_cls

    def __init__(
        self,
        obs_space: Box,
        spec: StateMLP.spec_cls,
        params_fn: Callable[[int], nn.Module],
        dist: ptd.ConditionalDistribution,
    ):
        encoder = StateMLP(obs_space, spec)
        params = params_fn(encoder.out_features)
        params_module = nn.Sequential(encoder, params)
        super().__init__(params_module, dist)

        self.encoder = encoder
        self.spec = spec

    def initialize_parameters(self, initializer_spec: dict):
        """Initialize all Linear models in the encoder.

        Args:
            initializer_spec: Dictionary with mandatory `name` key corresponding
                to the initializer function name in `torch.nn.init` and optional
                keyword arguments.
        """
        self.encoder.initialize_parameters(initializer_spec)


class MLPContinuousPolicy(MLPStochasticPolicy):
    """Multilayer perceptron policy for continuous actions.

    Args:
        obs_space: Observation space
        action_space: Action space
        mlp_spec: Specifications for the multilayer perceptron
        input_dependent_scale: Whether to parameterize the Gaussian standard
            deviation as a function of the state
    """

    def __init__(
        self,
        obs_space: Box,
        action_space: Box,
        mlp_spec: MLPStochasticPolicy.spec_cls,
        input_dependent_scale: bool,
    ):
        def params_fn(out_features):
            return nnx.PolicyNormalParams(
                out_features,
                action_space.shape[0],
                input_dependent_scale=input_dependent_scale,
            )

        dist = ptd.TransformedDistribution(
            ptd.Independent(ptd.Normal(), reinterpreted_batch_ndims=1),
            ptd.flows.TanhSquashTransform(
                low=torch.as_tensor(action_space.low),
                high=torch.as_tensor(action_space.high),
                event_dim=1,
            ),
        )
        super().__init__(obs_space, mlp_spec, params_fn, dist)


class MLPDiscretePolicy(MLPStochasticPolicy):
    """Multilayer perceptron policy for discrete actions.

    Args:
        obs_space: Observation space
        action_space: Action space
        mlp_spec: Specifications for the multilayer perceptron
    """

    def __init__(
        self,
        obs_space: Box,
        action_space: Discrete,
        mlp_spec: MLPStochasticPolicy.spec_cls,
    ):
        def params_fn(out_features):
            return nnx.CategoricalParams(out_features, action_space.n)

        dist = ptd.Categorical()
        super().__init__(obs_space, mlp_spec, params_fn, dist)


class Alpha(nn.Module):
    """Wraps a single scalar coefficient parameter.

    Allows learning said coefficient by having it as a parameter

    Args:
        initial_alpha: Value to initialize the coefficient to

    Attributes:
        lob_alpha: Natural logarithm of the current coefficient
    """

    def __init__(self, initial_alpha: float):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.empty([]).fill_(initial_alpha).log())

    def forward(self):  # pylint:disable=arguments-differ
        return self.log_alpha.exp()
