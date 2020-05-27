"""Losses for Importance Sampled Fitted V Iteration."""
import torch
from ray.rllib import SampleBatch
from ray.rllib.utils import override

import raylab.utils.dictionaries as dutil


class ISFittedVIteration:
    """Loss function for Importance Sampled Fitted V Iteration.

    Args:
        critic (callable): state-value function
        target_critic (callable): state-value function for the next state
        gamma (float): discount factor
    """

    IS_RATIOS = "is_ratios"

    def __init__(self, critic, target_critic, **config):
        self.critic = critic
        self.target_critic = target_critic
        self.config = config

    def __call__(self, batch):
        """Compute loss for importance sampled fitted V iteration."""
        obs, is_ratios = dutil.get_keys(batch, SampleBatch.CUR_OBS, self.IS_RATIOS)

        values = self.critic(obs).squeeze(-1)
        with torch.no_grad():
            targets = self.sampled_one_step_state_values(batch)
        value_loss = torch.mean(
            is_ratios * torch.nn.MSELoss(reduction="none")(values, targets) / 2
        )
        return value_loss, {"loss(critic)": value_loss.item()}

    def sampled_one_step_state_values(self, batch):
        """Bootstrapped approximation of true state-value using sampled transition."""
        next_obs, rewards, dones = dutil.get_keys(
            batch, SampleBatch.NEXT_OBS, SampleBatch.REWARDS, SampleBatch.DONES,
        )
        return torch.where(
            dones,
            rewards,
            rewards + self.config["gamma"] * self.target_critic(next_obs).squeeze(-1),
        )


class ISSoftVIteration(ISFittedVIteration):
    """Loss function for Importance Sampled Soft V Iteration.

    Args:
        critic (callable): state-value function
        target_critic (callable): state-value function for the next state
        actor (callable): stochastic policy
        alpha (callable): entropy coefficient schedule
        gamma (float): discount factor
    """

    # pylint:disable=too-few-public-methods
    ENTROPY = "entropy"

    def __init__(self, critic, target_critic, actor, alpha, **config):
        super().__init__(critic, target_critic, **config)
        self.actor = actor
        self.alpha = alpha

    @override(ISFittedVIteration)
    def sampled_one_step_state_values(self, batch):
        """Bootstrapped approximation of true state-value using sampled transition."""
        if self.ENTROPY in batch:
            entropy = batch[self.ENTROPY]
        else:
            with torch.no_grad():
                _, logp = self.actor(batch[SampleBatch.CUR_OBS])
                entropy = -logp

        next_obs, rewards, dones = dutil.get_keys(
            batch, SampleBatch.NEXT_OBS, SampleBatch.REWARDS, SampleBatch.DONES,
        )
        gamma = self.config["gamma"]
        augmented_rewards = rewards + self.alpha() * entropy
        return torch.where(
            dones,
            augmented_rewards,
            augmented_rewards + gamma * self.target_critic(next_obs).squeeze(-1),
        )
