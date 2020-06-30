"""Network and configurations for modules with stochastic model ensembles."""
from typing import List

import torch
import torch.nn as nn

from .single import StochasticModel


class StochasticModelEnsemble(nn.ModuleList):
    """A static list of stochastic dynamics models.

    Args:
        models: List of StochasticModel modules
    """

    # pylint:disable=abstract-method

    def __init__(self, models: List[StochasticModel]):
        cls_name = type(self).__name__
        assert all(
            isinstance(m, StochasticModel) for m in models
        ), f"All modules in {cls_name} must be instances of StochasticModel."
        super().__init__(models)

    @torch.jit.export
    def sample(self, obs, action, sample_shape: List[int] = ()):
        """Compute samples and likelihoods for each model in the ensemble."""
        outputs = [m.sample(obs, action, sample_shape) for m in self]
        sample = torch.stack([s for s, _ in outputs])
        logp = torch.stack([p for _, p in outputs])
        return sample, logp

    @torch.jit.export
    def rsample(self, obs, action, sample_shape: List[int] = ()):
        """Compute reparemeterized samples and likelihoods for each model."""
        outputs = [m.rsample(obs, action, sample_shape) for m in self]
        sample = torch.stack([s for s, _ in outputs])
        logp = torch.stack([p for _, p in outputs])
        return sample, logp

    @torch.jit.export
    def log_prob(self, obs, action, next_obs):
        """Compute likelihoods for each model in the ensemble."""
        return torch.stack([m.log_prob(obs, action, next_obs) for m in self])


class ForkedStochasticModelEnsemble(StochasticModelEnsemble):
    """Ensemble of stochastic models with parallelized methods."""

    # pylint:disable=abstract-method,protected-access

    @torch.jit.export
    def sample(self, obs, action, sample_shape: List[int] = ()):
        futures = [torch.jit._fork(m.sample, obs, action, sample_shape) for m in self]
        outputs = [torch.jit._wait(f) for f in futures]
        sample = torch.stack([s for s, _ in outputs])
        logp = torch.stack([p for _, p in outputs])
        return sample, logp

    @torch.jit.export
    def rsample(self, obs, action, sample_shape: List[int] = ()):
        futures = [torch.jit._fork(m.rsample, obs, action, sample_shape) for m in self]
        outputs = [torch.jit._wait(f) for f in futures]
        sample = torch.stack([s for s, _ in outputs])
        logp = torch.stack([p for _, p in outputs])
        return sample, logp

    @torch.jit.export
    def log_prob(self, obs, action, next_obs):
        futures = [torch.jit._fork(m.log_prob, obs, action, next_obs) for m in self]
        return torch.stack([torch.jit._wait(f) for f in futures])
