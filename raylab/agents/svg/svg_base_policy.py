"""Base Policy with common methods for all SVG variations."""
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

from raylab.envs.rewards import get_reward_fn
from raylab.policy import TorchPolicy, TargetNetworksMixin


class SVGBaseTorchPolicy(TargetNetworksMixin, TorchPolicy):
    """Stochastic Value Gradients policy using PyTorch."""

    # pylint: disable=abstract-method

    IS_RATIOS = "is_ratios"

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.reward = get_reward_fn(self.config["env"], self.config["env_config"])

    @torch.no_grad()
    def add_importance_sampling_ratios(self, batch_tensors):
        """Compute and add truncated importance sampling ratios to tensor batch."""
        is_ratios = self._compute_is_ratios(batch_tensors)
        _is_ratios = torch.clamp(is_ratios, max=self.config["max_is_ratio"])
        batch_tensors[self.IS_RATIOS] = _is_ratios
        return batch_tensors, {"avg_is_ratio": is_ratios.mean().item()}

    def _compute_is_ratios(self, batch_tensors):
        curr_logp = self.module.actor.log_prob(
            batch_tensors[SampleBatch.CUR_OBS], batch_tensors[SampleBatch.ACTIONS]
        )
        is_ratio = torch.exp(curr_logp - batch_tensors[SampleBatch.ACTION_LOGP])
        return is_ratio

    def compute_joint_model_value_loss(self, batch_tensors):
        """Compute model MLE loss and fitted value function loss."""
        mle_loss = self._avg_model_logp(batch_tensors).neg()

        with torch.no_grad():
            targets = self._compute_value_targets(batch_tensors)
        _is_ratios = batch_tensors[self.IS_RATIOS]
        values = self.module.critic(batch_tensors[SampleBatch.CUR_OBS]).squeeze(-1)
        value_loss = torch.mean(
            _is_ratios * nn.MSELoss(reduction="none")(values, targets) / 2
        )

        loss = mle_loss + self.config["vf_loss_coeff"] * value_loss
        return loss, {"mle_loss": mle_loss.item(), "value_loss": value_loss.item()}

    def _avg_model_logp(self, batch_tensors):
        return self.module.model.log_prob(
            batch_tensors[SampleBatch.CUR_OBS],
            batch_tensors[SampleBatch.ACTIONS],
            batch_tensors[SampleBatch.NEXT_OBS],
        ).mean()

    def _compute_value_targets(self, batch_tensors):
        next_obs = batch_tensors[SampleBatch.NEXT_OBS]
        next_vals = self.module.target_critic(next_obs).squeeze(-1)
        targets = torch.where(
            batch_tensors[SampleBatch.DONES],
            batch_tensors[SampleBatch.REWARDS],
            batch_tensors[SampleBatch.REWARDS] + self.config["gamma"] * next_vals,
        )
        return targets
