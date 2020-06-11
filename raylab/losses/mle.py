"""Loss functions for Maximum Likelihood Estimation."""
from typing import Dict
from typing import List
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.modules.mixins.stochastic_model_mixin import StochasticModel
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

    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Compute Maximum Likelihood Estimation (MLE) model loss.

        Returns:
            A tuple containg a 0d loss tensor and a dictionary of loss
            statistics
        """
        obs, actions, next_obs = get_keys(batch, *self.batch_keys)
        loss = -self.model_likelihood(obs, actions, next_obs).mean()
        return loss, {"loss(model)": loss.item()}

    def model_likelihood(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor
    ) -> Tensor:
        """Compute likelihood of a transition under the model."""
        return self.model.log_prob(obs, actions, next_obs)


class ModelEnsembleMLE(Loss):
    """MLE loss function for ensemble of models.

    Args:
        models: the list of models
    """

    batch_keys: Tuple[str, str, str] = MaximumLikelihood.batch_keys

    def __init__(self, models: List[StochasticModel]):
        self.models = models

    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Compute Maximum Likelihood Estimation (MLE) loss for each model.

        Returns:
            A tuple with a 1d loss tensor containing each model's loss and a
            dictionary of loss statistics
        """
        obs, actions, next_obs = get_keys(batch, *self.batch_keys)
        logps = self.model_likelihoods(obs, actions, next_obs)
        loss = -torch.stack(logps)
        info = {f"loss(models[{i}])": -l.item() for i, l in enumerate(logps)}
        return loss, info

    def model_likelihoods(
        self, obs: Tensor, actions: Tensor, next_obs: Tensor
    ) -> List[Tensor]:
        """Compute transition likelihood under each model."""
        return [m.log_prob(obs, actions, next_obs).mean() for m in self.models]
