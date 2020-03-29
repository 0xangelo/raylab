"""Base Policy with common methods for all SVG variations."""
import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

from raylab.policy import TorchPolicy, AdaptiveKLCoeffMixin, TargetNetworksMixin
from raylab.modules.catalog import get_module


class SVGBaseTorchPolicy(AdaptiveKLCoeffMixin, TargetNetworksMixin, TorchPolicy):
    """Stochastic Value Gradients policy using PyTorch."""

    # pylint: disable=abstract-method

    ACTION_LOGP = "action_logp"
    IS_RATIOS = "is_ratios"

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.reward = None

    def set_reward_fn(self, reward_fn):
        """Set the reward function to use when unrolling the policy and model."""

    @torch.no_grad()
    @override(TorchPolicy)
    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        # pylint: disable=too-many-arguments,unused-argument
        obs_batch = self.convert_to_tensor(obs_batch)
        actions, logp = self.module.actor.rsample(obs_batch)

        extra_fetches = {self.ACTION_LOGP: logp.cpu().numpy()}
        return actions.cpu().numpy(), state_batches, extra_fetches

    @override(TorchPolicy)
    def make_module(self, obs_space, action_space, config):
        module_config = config["module"]
        module = get_module(
            module_config["name"], obs_space, action_space, module_config
        )
        return torch.jit.script(module) if module_config["torch_script"] else module

    @torch.no_grad()
    @override(AdaptiveKLCoeffMixin)
    def _kl_divergence(self, sample_batch):
        return self._avg_kl_divergence(self._lazy_tensor_dict(sample_batch)).item()

    def _avg_kl_divergence(self, batch_tensors):
        """Compute the empirical average KL divergence given sample tensors."""

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
        is_ratio = torch.exp(curr_logp - batch_tensors[self.ACTION_LOGP])
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
        return self.module.model.logp(
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
