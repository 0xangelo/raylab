"""Loss functions for Maximum Likelihood Estimation."""
from ray.rllib import SampleBatch
from raylab.utils.dictionaries import get_keys


class MaximumLikelihood:
    """Loss function for model learning of single transitions."""

    def __init__(self, model):
        self.model = model

    def __call__(self, batch):
        """Compute Maximum Likelihood Estimation (MLE) model loss."""
        obs, actions, next_obs = get_keys(
            batch, SampleBatch.CUR_OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS
        )
        loss = -self.model_likelihood(obs, actions, next_obs).mean()
        return loss, {"loss(mle)": loss.item()}

    def model_likelihood(self, obs, actions, next_obs):
        """Compute likelihood of a transition under the model."""
        return self.model.log_prob(obs, actions, next_obs)
