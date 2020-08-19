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
        self.tag = "nll"

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
            nll + reg for nll, reg in zip(nlls, self.regularizations(dist_params))
        ]

        info = {f"{self.tag}(models[{i}])": n.item() for i, n in enumerate(nlls)}
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
    def regularizations(cls, params_list: List[TensorDict]) -> List[Tensor]:
        """Computes logvar bound penalties if needed.

        Args:
            params_list: List of model distribution parameters

        Returns:
            List of regularization factors for each model's loss
        """
        return list(map(cls.regularize_if_needed, params_list))

    @staticmethod
    def regularize_if_needed(params: TensorDict) -> Tensor:
        """Add logvar bound penalty if needed.

        Args:
            params: Model distribution parameters

        Returns:
            Regularization factors for model loss
        """
        should_regularize = (
            "max_logvar" in params
            and params["max_logvar"].requires_grad
            and "min_logvar" in params
            and params["min_logvar"].requires_grad
        )
        if should_regularize:
            return 0.01 * params["max_logvar"].sum() - 0.01 * params["min_logvar"].sum()

        return torch.zeros([])
