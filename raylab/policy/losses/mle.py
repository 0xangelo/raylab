"""Loss functions for Maximum Likelihood Estimation."""
from typing import List
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.policy.modules.model.stochastic.ensemble import SME
from raylab.policy.modules.model.stochastic.single import StochasticModel
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict
from raylab.utils.dictionaries import get_keys

from .abstract import Loss


class MaximumLikelihood(Loss):
    """Loss function for model learning of single transitions.

    Args:
        model: parametric stochastic model
    """

    batch_keys: Tuple[str, str, str] = (
        SampleBatch.CUR_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.NEXT_OBS,
    )

    def __init__(self, model: StochasticModel):
        self.model = model

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        """Compute Maximum Likelihood Estimation (MLE) model loss.

        Returns:
            A tuple containg a 0d loss tensor and a dictionary of loss
            statistics
        """
        obs, actions, next_obs = get_keys(batch, *self.batch_keys)

        dist_params = self.model(obs, actions)
        loss = -self.model.dist.log_prob(next_obs, dist_params).mean()
        if "max_logvar" in dist_params and "min_logvar" in dist_params:
            loss += 0.01 * dist_params["max_logvar"].sum()
            loss += -0.01 * dist_params["min_logvar"].sum()

        return loss, {"loss(model)": loss.item()}


class ModelEnsembleMLE(Loss):
    """MLE loss function for ensemble of models.

    Args:
        models: the list of models
    """

    batch_keys: Tuple[str, str, str] = MaximumLikelihood.batch_keys

    def __init__(self, models: SME):
        self.models = models
        self.tag = "MLE"

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        """Compute Maximum Likelihood Estimation (MLE) loss for each model.

        Returns:
            A tuple with a 1d loss tensor containing each model's loss and a
            dictionary of loss statistics
        """
        obs, actions, next_obs = map(
            self.expand_foreach_model, get_keys(batch, *self.batch_keys)
        )

        dist_params = self.models(obs, actions)
        losses = [
            -logp.mean()
            for logp in self.models.log_prob_from_params(next_obs, dist_params)
        ]
        losses_reg = [
            nll + 0.01 * p["max_logvar"].sum() - 0.01 * p["min_logvar"].sum()
            if "max_logvar" in p and "min_logvar" in p
            else nll
            for nll, p in zip(losses, dist_params)
        ]

        info = {f"{self.tag}(models[{i}])": n.item() for i, n in enumerate(losses_reg)}
        return torch.stack(losses_reg), info

    def expand_foreach_model(self, tensor: Tensor) -> List[Tensor]:
        """Add first dimension to tensor with the size of the model ensemble.

        Args:
            tensor: Tensor of shape `S`

        Returns:
            Tensor `tensor` expanded to shape `(N,) + S`
        """
        return [tensor.clone() for _ in range(len(self.models))]

    def compile(self):
        self.models = torch.jit.script(self.models)
