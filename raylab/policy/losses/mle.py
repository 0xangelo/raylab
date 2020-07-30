"""Loss functions for Maximum Likelihood Estimation."""
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.policy.modules.model.stochastic.ensemble import StochasticModelEnsemble
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

    def __init__(self, models: StochasticModelEnsemble):
        self.models = models

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
        loss = -self.models.dist_log_prob(next_obs, dist_params).mean(dim=-1)
        if "max_logvar" in dist_params and "min_logvar" in dist_params:
            loss += 0.01 * dist_params["max_logvar"].sum()
            loss += -0.01 * dist_params["min_logvar"].sum()

        info = {f"loss(models[{i}])": s for i, s in enumerate(loss.tolist())}
        return loss, info

    def expand_foreach_model(self, tensor: Tensor) -> Tensor:
        """Add first dimension to tensor with the size of the model ensemble.

        Args:
            tensor: Tensor of shape `S`

        Returns:
            Tensor `tensor` expanded to shape `(N,) + S`
        """
        return tensor.expand((len(self.models),) + tensor.shape)

    def compile(self):
        self.models = torch.jit.script(self.models)
