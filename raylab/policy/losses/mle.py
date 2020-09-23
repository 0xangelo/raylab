"""Loss functions for Maximum Likelihood Estimation."""
from typing import List
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from torch import Tensor
from torch.jit import fork
from torch.jit import wait

from raylab.policy.modules.model import ForkedSME
from raylab.policy.modules.model import SME
from raylab.policy.modules.model import StochasticModel
from raylab.utils.dictionaries import get_keys
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict

from .abstract import Loss


class LogVarReg(nn.Module):
    """Compute logvar bound penalty if needed."""

    # pylint:disable=abstract-method
    def forward(self, params: TensorDict) -> Tensor:
        """Compute logvar bound penalty if needed.

        Args:
            params: Model distribution parameters

        Returns:
            Regularization factors for model loss
        """
        # pylint:disable=no-self-use
        should_regularize = (
            "max_logvar" in params
            and params["max_logvar"].requires_grad
            and "min_logvar" in params
            and params["min_logvar"].requires_grad
        )
        if should_regularize:
            return 0.01 * params["max_logvar"].sum() - 0.01 * params["min_logvar"].sum()

        return torch.zeros(1).squeeze()


class NLLLoss(nn.Module):
    """Compute Negative Log-Likelihood loss."""

    # pylint:disable=abstract-method

    def __init__(self, model: StochasticModel):
        super().__init__()
        self.model = model
        self.logvar_reg = LogVarReg()

    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> Tensor:
        # pylint:disable=missing-function-docstring
        params = self.model(obs, act)
        logp = self.model.log_prob(new_obs, params)
        nll = -logp.mean()
        regularizer = self.logvar_reg(params)
        return nll + regularizer


class Losses(nn.ModuleList):
    # pylint:disable=abstract-method,missing-class-docstring
    def __init__(self, losses: List[NLLLoss]):
        assert all(isinstance(loss, NLLLoss) for loss in losses)
        super().__init__(losses)

    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> List[Tensor]:
        # pylint:disable=arguments-differ
        return [loss(obs, act, new_obs) for loss in self]


class ForkedLosses(Losses):
    # pylint:disable=abstract-method,missing-class-docstring
    def forward(self, obs: Tensor, act: Tensor, new_obs: Tensor) -> List[Tensor]:
        futures = [fork(loss, obs, act, new_obs) for loss in self]
        return [wait(f) for f in futures]


class MaximumLikelihood(Loss):
    """Loss function for model learning of single transitions.

    Args:
        models: parametric stochastic model (single or ensemble)
    """

    batch_keys: Tuple[str, str, str] = (
        SampleBatch.CUR_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.NEXT_OBS,
    )
    _last_output: Tuple[Tensor, StatDict]

    def __init__(self, models: Union[StochasticModel, SME]):
        if isinstance(models, StochasticModel):
            # Treat everything as if ensemble
            models = SME([models])
        self.models = models
        self.tag = "nll"
        self.build_losses()
        self._last_losses = torch.zeros(len(self.models))

    def build_losses(self):
        # pylint:disable=missing-function-docstring
        models = self.models
        losses = [NLLLoss(m) for m in models]
        cls = ForkedLosses if isinstance(models, ForkedSME) else Losses
        self.loss_fns = cls(losses)

    @property
    def last_output(self) -> Tuple[Tensor, StatDict]:
        """Last computed losses for each individual model and associated info."""
        return self._last_output

    def compile(self):
        # pylint:disable=missing-function-docstring,attribute-defined-outside-init
        self.loss_fns = torch.jit.script(self.loss_fns)

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        """Compute Maximum Likelihood Estimation (MLE) model loss.

        Returns:
            A tuple with a 1d loss tensor containing each model's loss and a
            dictionary of loss statistics
        """
        obs, act, new_obs = get_keys(batch, *self.batch_keys)
        nlls = self.loss_fns(obs, act, new_obs)

        losses = torch.stack(nlls)
        info = {f"{self.tag}(models[{i}])": n.item() for i, n in enumerate(nlls)}
        self._last_output = (losses, info)
        return losses.mean(), info
