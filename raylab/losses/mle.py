"""Loss functions for Maximum Likelihood Estimation."""
import torch
from ray.rllib import SampleBatch
from ray.rllib.utils import override

from raylab.utils.dictionaries import get_keys


class MaximumLikelihood:
    """Loss function for model learning of single transitions."""

    batch_keys = (SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS)

    def __init__(self, model):
        self.model = model

    def __call__(self, batch):
        """Compute Maximum Likelihood Estimation (MLE) model loss."""
        obs, actions, next_obs = get_keys(batch, *self.batch_keys)
        loss = -self.model_likelihood(obs, actions, next_obs).mean()
        return loss, {"loss(model)": loss.item()}

    def model_likelihood(self, obs, actions, next_obs):
        """Compute likelihood of a transition under the model."""
        return self.model.log_prob(obs, actions, next_obs)


class ModelEnsembleMLE(MaximumLikelihood):
    """MLE loss function for ensemble of models."""

    # pylint:disable=too-few-public-methods

    @override(MaximumLikelihood)
    def __call__(self, batch):
        obs, actions, next_obs = get_keys(batch, *self.batch_keys)
        logps = self.model_likelihoods(obs, actions, next_obs)
        loss = -torch.stack(logps).mean()
        info = {f"loss(models[{i}])": -l.item() for i, l in enumerate(logps)}
        info["loss(models)"] = loss.item()
        return loss, info

    def model_likelihoods(self, obs, actions, next_obs):
        """Compute transition likelihood under each model."""
        return [m.log_prob(obs, actions, next_obs).mean() for m in self.model]
