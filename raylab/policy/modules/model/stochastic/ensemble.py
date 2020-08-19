"""Network and configurations for modules with stochastic model ensembles."""
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.jit import fork
from torch.jit import wait

from raylab.utils.annotations import TensorDict

from .single import StochasticModel

SampleLogp = Tuple[Tensor, Tensor]


class SME(nn.ModuleList):
    """Stochastic Model Ensemble.

    A static NN module list of `N` stochastic dynamics models. Implements the
    StochasticModel API but returns python lists of `N` outputs, one for each
    model in the ensemble.

    Assumes inputs are lists of the same length as the model ensemble.
    Applies each model in the ensemble to one of the inputs in the list.

    Args:
        models: List of StochasticModel modules

    Notes:
        `O` is the observation shape and `A` is the action shape.
    """

    # pylint:disable=abstract-method

    def __init__(self, models: List[StochasticModel]):
        cls_name = type(self).__name__
        assert all(
            isinstance(m, StochasticModel) for m in models
        ), f"All modules in {cls_name} must be instances of StochasticModel."
        super().__init__(models)

    def forward(self, obs: List[Tensor], act: List[Tensor]) -> List[TensorDict]:
        # pylint:disable=arguments-differ
        return [m(obs[i], act[i]) for i, m in enumerate(self)]

    @torch.jit.export
    def sample(self, params: List[TensorDict]) -> List[SampleLogp]:
        """Compute samples and likelihoods for each model in the ensemble.

        Args:
            obs: List of `N` observation tensors of shape `(*,) + O`
            action: List of `N` action tensors of shape `(*,) + A`
            sample_shape: Sample shape argument for each model in the ensemble

        Returns:
           List of `N` tuples of sample and log-likelihood tensors of shape
           `S + (*,) + O` and `S + (*,)` respectively, where `S` is the
           `sample_shape`.
        """
        return [m.sample(params[i]) for i, m in enumerate(self)]

    @torch.jit.export
    def rsample(self, params: List[TensorDict]) -> List[SampleLogp]:
        """Compute reparameterized samples and likelihoods for each model.

        Uses the same semantics as :meth:`SME.sample`.
        """
        return [m.rsample(params[i]) for i, m in enumerate(self)]

    @torch.jit.export
    def log_prob(self, new_obs: List[Tensor], params: List[TensorDict]) -> List[Tensor]:
        """Compute likelihoods for each model in the ensemble.

        Args:
            obs: List of `N` observation tensors of shape `(*,) + O`
            action: List of `N` action tensors of shape `(*,) + A`
            new_obs: List of `N` observation tensors of shape `(*,) + O`

        Returns:
           List of `N` log-likelihood tensors of shape `(*,)`
        """
        return [m.log_prob(new_obs[i], params[i]) for i, m in enumerate(self)]

    @torch.jit.export
    def deterministic(self, params: List[TensorDict]) -> List[SampleLogp]:
        """Compute deterministic new observations and their likelihoods for each model.

        Uses the same semantics as :meth:`SME.sample`.
        """
        return [m.deterministic(params[i]) for i, m in enumerate(self)]


class ForkedSME(SME):
    """Stochastic Model Ensemble with parallelized methods."""

    # pylint:disable=abstract-method

    def forward(self, obs: List[Tensor], act: List[Tensor]) -> List[TensorDict]:
        futures = [fork(m, obs[i], act[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]

    @torch.jit.export
    def sample(self, params: List[TensorDict]) -> List[SampleLogp]:
        futures = [fork(m.sample, params[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]

    @torch.jit.export
    def rsample(self, params: List[TensorDict]) -> List[SampleLogp]:
        futures = [fork(m.rsample, params[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]

    @torch.jit.export
    def log_prob(self, new_obs: List[Tensor], params: List[TensorDict]) -> List[Tensor]:
        futures = [fork(m.log_prob, new_obs[i], params[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]

    @torch.jit.export
    def deterministic(self, params: List[TensorDict]) -> List[SampleLogp]:
        futures = [fork(m.deterministic, params[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]
