"""Base Policy with common methods for all SVG variations."""
import torch
from ray.rllib import SampleBatch

from raylab.policy import EnvFnMixin
from raylab.policy import TorchPolicy
from raylab.policy.action_dist import WrapStochasticPolicy
from raylab.policy.losses import ISFittedVIteration
from raylab.policy.losses import MaximumLikelihood
from raylab.pytorch.nn.utils import update_polyak


class SVGTorchPolicy(EnvFnMixin, TorchPolicy):
    """Stochastic Value Gradients policy using PyTorch."""

    # pylint:disable=abstract-method
    dist_class = WrapStochasticPolicy

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.loss_model = MaximumLikelihood(self.module.model)
        self.loss_critic = ISFittedVIteration(
            self.module.critic, self.module.target_critic
        )
        self.loss_critic.gamma = self.config["gamma"]

    @torch.no_grad()
    def add_truncated_importance_sampling_ratios(self, batch_tensors):
        """Compute and add truncated importance sampling ratios to tensor batch."""
        is_ratios = self.importance_sampling_ratios(batch_tensors)
        _is_ratios = torch.clamp(is_ratios, max=self.config["max_is_ratio"])
        batch_tensors[ISFittedVIteration.IS_RATIOS] = _is_ratios
        return batch_tensors, {"is_ratio_mean": is_ratios.mean().item()}

    def importance_sampling_ratios(self, batch_tensors):
        """Compute unrestricted importance sampling ratios."""
        curr_logp = self.module.actor.log_prob(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        is_ratio = torch.exp(curr_logp - batch_tensors[SampleBatch.ACTION_LOGP])
        return is_ratio

    def compute_joint_model_value_loss(self, batch_tensors):
        """Compute model MLE loss and fitted value function loss."""
        mle_loss, mle_info = self.loss_model(batch_tensors)
        isfv_loss, isfv_info = self.loss_critic(batch_tensors)

        loss = mle_loss + self.config["vf_loss_coeff"] * isfv_loss
        return loss, {**mle_info, **isfv_info}

    def _update_polyak(self):
        update_polyak(
            self.module.critic, self.module.target_critic, self.config["polyak"]
        )
