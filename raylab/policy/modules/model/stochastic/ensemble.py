"""Network and configurations for modules with stochastic model ensembles."""
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

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
    def sample(
        self, obs: Tensor, action: Tensor, sample_shape: List[int] = ()
    ) -> Tuple[Tensor, Tensor]:
        """Compute samples and likelihoods for each model in the ensemble.

        Splits inputs by the first dimension in `N` chunks, where `N` is the
        ensemble size. Then, applies each model in the ensemble to one of the
        slices.

        `O` is the observation shape and `A` is the action shape.

        Args:
            obs: Observation tensor of shape `(N, *) + O`
            action: Action tensor of shape `(N, *) + A`
            sample_shape: Shape to append to the samples of each model in the
                ensemble. Samples from all the models are then concatenated.

        Returns:
           Sample and log-likelihood tensors of shape `(N,) + S + (*,) + O` and
           `(N,) + S + (*,)` respectively, where `S` is the `sample_shape`.
        """
        outputs = [
            m.sample(obs[i], action[i], sample_shape) for i, m in enumerate(self)
        ]
        sample = torch.stack([s for s, _ in outputs])
        logp = torch.stack([p for _, p in outputs])
        return sample, logp

    @torch.jit.export
    def rsample(
        self, obs, action, sample_shape: List[int] = ()
    ) -> Tuple[Tensor, Tensor]:
        """Compute reparameterized samples and likelihoods for each model.

        Uses the same semantics as :meth:`StochasticModelEnsemble.sample`.
        """
        outputs = [
            m.rsample(obs[i], action[i], sample_shape) for i, m in enumerate(self)
        ]
        sample = torch.stack([s for s, _ in outputs])
        logp = torch.stack([p for _, p in outputs])
        return sample, logp

    @torch.jit.export
    def log_prob(self, obs, action, next_obs) -> Tensor:
        """Compute likelihoods for each model in the ensemble.

        Splits inputs by the first dimension in `N` chunks, where `N` is the
        ensemble size. Then, applies each model in the ensemble to one of the
        slices.

        Args:
            obs: Observation tensor of shape `(N, *) + O`
            action: Action tensor of shape `(N, *) + A`
            next_obs: Observation tensor of shape `(N, *) + O`

        Returns:
           Log-likelihood tensor of shape `(N, *)`
        """
        return torch.stack(
            [m.log_prob(obs[i], action[i], next_obs[i]) for i, m in enumerate(self)]
        )


class ForkedStochasticModelEnsemble(StochasticModelEnsemble):
    """Ensemble of stochastic models with parallelized methods."""

    # pylint:disable=abstract-method,protected-access

    @torch.jit.export
    def sample(self, obs, action, sample_shape: List[int] = ()):
        futures = [
            torch.jit._fork(m.sample, obs[i], action[i], sample_shape)
            for i, m in enumerate(self)
        ]
        outputs = [torch.jit._wait(f) for f in futures]
        sample = torch.stack([s for s, _ in outputs])
        logp = torch.stack([p for _, p in outputs])
        return sample, logp

    @torch.jit.export
    def rsample(self, obs, action, sample_shape: List[int] = ()):
        futures = [
            torch.jit._fork(m.rsample, obs[i], action[i], sample_shape)
            for i, m in enumerate(self)
        ]
        outputs = [torch.jit._wait(f) for f in futures]
        sample = torch.stack([s for s, _ in outputs])
        logp = torch.stack([p for _, p in outputs])
        return sample, logp

    @torch.jit.export
    def log_prob(self, obs, action, next_obs):
        futures = [
            torch.jit._fork(m.log_prob, obs[i], action[i], next_obs[i])
            for i, m in enumerate(self)
        ]
        return torch.stack([torch.jit._wait(f) for f in futures])
