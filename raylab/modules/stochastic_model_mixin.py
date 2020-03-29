"""Support for modules with stochastic policies."""
from typing import List

import torch
import torch.nn as nn
from ray.rllib.utils.annotations import override

from .basic import StateActionEncoder, NormalParams
from .distributions import Independent, Normal


class StochasticModelMixin:
    """Adds constructor for modules with stochastic dynamics model."""

    # pylint:disable=too-few-public-methods

    def _make_model(self, obs_space, action_space, config):
        model = nn.ModuleDict()
        model_config = config["model"]

        params_module = self._make_model_encoder(obs_space, action_space, model_config)
        dist_module = Independent(Normal(), reinterpreted_batch_ndims=1)

        if model_config["residual"]:
            model = ResidualStochasticModel(params_module, dist_module)
        else:
            model = StochasticModel(params_module, dist_module)

        return {"model": model}

    @staticmethod
    def _make_model_encoder(obs_space, action_space, config):
        return GaussianDynamicsParams(obs_space, action_space, config)


class GaussianDynamicsParams(nn.Module):
    """
    Neural network module mapping inputs to Normal distribution parameters.
    """

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.logits = StateActionEncoder(
            obs_dim=obs_space.shape[0],
            action_dim=action_space.shape[0],
            units=config["units"],
            delay_action=config["delay_action"],
            activation=config["activation"],
            **config["initializer_options"],
        )
        self.params = NormalParams(
            self.logits.out_features,
            obs_space.shape[0],
            input_dependent_scale=config["input_dependent_scale"],
        )

    @override(nn.Module)
    def forward(self, obs, actions):  # pylint:disable=arguments-differ
        return self.params(self.logits(obs, actions))


class StochasticModel(nn.Module):
    """Represents a stochastic model as a conditional distribution module."""

    def __init__(self, params_module, dist_module):
        super().__init__()
        self.params = params_module
        self.dist = dist_module

    @override(nn.Module)
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
        return self.dist.rsample(params, sample_shape)

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
        return self.dist.log_prob(params, next_obs)

    @torch.jit.export
    def cdf(self, obs, action, next_obs):
        """Returns the cumulative density/mass function evaluated at `next_obs`."""
        params = self(obs, action)
        return self.dist.cdf(params, next_obs)

    @torch.jit.export
    def icdf(self, obs, action, prob):
        """Returns the inverse cumulative density/mass function evaluated at `prob`."""
        params = self(obs, action)
        return self.dist.icdf(params, prob)

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
        return self.dist.reproduce(params, next_obs)


class ResidualStochasticModel(StochasticModel):
    """
    Represents a stochastic model as a conditional distribution module that predicts
    residuals.
    """

    @override(StochasticModel)
    @torch.jit.export
    def sample(self, obs, action, sample_shape: List[int] = ()):
        params = self(obs, action)
        res, log_prob = self.dist.sample(params, sample_shape)
        return obs + res, log_prob

    @override(StochasticModel)
    @torch.jit.export
    def rsample(self, obs, action, sample_shape: List[int] = ()):
        params = self(obs, action)
        res, log_prob = self.dist.rsample(params, sample_shape)
        return obs + res, log_prob

    @override(StochasticModel)
    @torch.jit.export
    def log_prob(self, obs, action, next_obs):
        params = self(obs, action)
        return self.dist.log_prob(params, next_obs - obs)

    @override(StochasticModel)
    @torch.jit.export
    def cdf(self, obs, action, next_obs):
        params = self(obs, action)
        return self.dist.cdf(params, next_obs - obs)

    @override(StochasticModel)
    @torch.jit.export
    def icdf(self, obs, action, prob):
        params = self(obs, action)
        residual = self.dist.icdf(params, prob)
        if residual is not None:
            return obs + residual
        return None

    @override(StochasticModel)
    @torch.jit.export
    def reproduce(self, obs, action, next_obs):
        params = self(obs, action)
        return obs + self.dist.reproduce(params, next_obs - obs)
