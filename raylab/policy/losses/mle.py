"""Loss functions for Maximum Likelihood Estimation."""
from typing import List
from typing import Tuple
from typing import Union

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
        models: parametric stochastic model (single or ensemble)
    """

    batch_keys: Tuple[str, str, str] = (
        SampleBatch.CUR_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.NEXT_OBS,
    )

    def __init__(self, models: Union[StochasticModel, SME]):
        if isinstance(models, StochasticModel):
            # Treat everything as if ensemble
            models = SME([models])
        self.models = models
        self.tag = "MLE"

    def compile(self):
        self.models = torch.jit.script(self.models)

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        """Compute Maximum Likelihood Estimation (MLE) model loss.

        Returns:
            A tuple with a 1d loss tensor containing each model's loss and a
            dictionary of loss statistics
        """
        obs, actions, next_obs = map(
            self.expand_foreach_model, get_keys(batch, *self.batch_keys)
        )

        dist_params = self.models(obs, actions)
        nlls = [-logp.mean() for logp in self.models.log_prob(next_obs, dist_params)]
        losses = [
            nll + reg for nll, reg in zip(nlls, self.add_regularizations(dist_params))
        ]

        info = {f"{self.tag}(models[{i}])": n.item() for i, n in enumerate(losses)}
        return torch.stack(losses), info

    def expand_foreach_model(self, tensor: Tensor) -> List[Tensor]:
        """Add first dimension to tensor with the size of the model ensemble.

        Args:
            tensor: Tensor of shape `S`

        Returns:
            Tensor `tensor` expanded to shape `(N,) + S`
        """
        return [tensor.clone() for _ in range(len(self.models))]

    @classmethod
    def add_regularizations(cls, params: List[TensorDict]) -> List[Tensor]:
        """Add logvar bound penalties if needed.

        Args:
            params: List of model distribution parameters

        Returns:
            List of regularization factors for each model's loss
        """
        return list(map(cls.regularize_if_needed, params))

    @staticmethod
    def regularize_if_needed(params: TensorDict) -> Tensor:
        """Add logvar bound penalty if needed.

        Args:
            params: Model distribution parameters

        Returns:
            Regularization factors for model loss
        """
        if "max_logvar" in params and "min_logvar" in params:
            return 0.01 * params["max_logvar"].sum() - 0.01 * params["min_logvar"].sum()

        return torch.zeros([])
