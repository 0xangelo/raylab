"""Support for modules with stochastic policies."""
from typing import List

import torch
import torch.nn as nn
from ray.rllib.utils import override

import raylab.pytorch.nn as nnx
import raylab.pytorch.nn.distributions as ptd
from raylab.pytorch.nn.init import initialize_
from raylab.utils.dictionaries import deep_merge


BASE_CONFIG = {
    "residual": True,
    "input_dependent_scale": False,
    # Number of independent models. If 0 return a single model accessible via the
    # 'model' atribute. If greater than 0, return a nn.ModuleList of models accessible
    # via the 'models' attribute.
    "ensemble_size": 0,
    "encoder": {"units": (32, 32), "activation": "ReLU", "delay_action": True},
    "initializer_options": {"name": "xavier_uniform"},
}


class StochasticModelMixin:
    """Adds constructor for modules with stochastic dynamics model."""

    # pylint:disable=too-few-public-methods

    def _make_model(self, obs_space, action_space, config):
        config = self.process_config(config)
        return self.build_models(obs_space, action_space, config)

    @staticmethod
    def process_config(config):
        """Fill in default configuration for models."""
        return deep_merge(BASE_CONFIG, config.get("model", {}), False, ["encoder"])

    def build_models(self, obs_space, action_space, config):
        """Decide whether to return a single model or an ensemble."""
        if config["ensemble_size"] == 0:
            return {"model": self.build_single_model(obs_space, action_space, config)}

        models = [
            self.build_single_model(obs_space, action_space, config)
            for _ in range(config["ensemble_size"])
        ]
        return {"models": nn.ModuleList(models)}

    @staticmethod
    def build_single_model(obs_space, action_space, config):
        """Build a stochastic model module."""
        params_module = GaussianDynamicsParams(obs_space, action_space, config)
        dist_module = ptd.Independent(ptd.Normal(), reinterpreted_batch_ndims=1)

        return StochasticModel.assemble(params_module, dist_module, config)


class GaussianDynamicsParams(nn.Module):
    """
    Neural network module mapping inputs to Normal distribution parameters.
    """

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        obs_size, act_size = obs_space.shape[0], action_space.shape[0]
        self.logits = nnx.StateActionEncoder(obs_size, act_size, **config["encoder"])
        self.logits.apply(
            initialize_(
                activation=config["encoder"].get("activation"),
                **config["initializer_options"]
            )
        )
        self.params = nnx.NormalParams(
            self.logits.out_features,
            obs_size,
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

    @classmethod
    def assemble(cls, params_module, dist_module, config):
        """Return a residual or normal stochastic model depending on configuration."""
        if config["residual"]:
            return ResidualStochasticModel(params_module, dist_module)
        return cls(params_module, dist_module)


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
        return self.dist.log_prob(next_obs - obs, params)

    @override(StochasticModel)
    @torch.jit.export
    def cdf(self, obs, action, next_obs):
        params = self(obs, action)
        return self.dist.cdf(next_obs - obs, params)

    @override(StochasticModel)
    @torch.jit.export
    def icdf(self, obs, action, prob):
        params = self(obs, action)
        residual = self.dist.icdf(prob, params)
        return obs + residual

    @override(StochasticModel)
    @torch.jit.export
    def reproduce(self, obs, action, next_obs):
        params = self(obs, action)
        sample_, log_prob_ = self.dist.reproduce(next_obs - obs, params)
        return obs + sample_, log_prob_
