"""Losses for Importance Sampled Fitted V Iteration."""
import torch
from ray.rllib import SampleBatch

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
        obs, is_ratios, next_obs, rewards, dones = dutil.get_keys(
            batch,
            SampleBatch.CUR_OBS,
            self.IS_RATIOS,
            SampleBatch.NEXT_OBS,
            SampleBatch.REWARDS,
            SampleBatch.DONES,
        )

        values = self.critic(obs).squeeze(-1)
        with torch.no_grad():
            targets = self.sampled_one_step_state_values(next_obs, rewards, dones)
        value_loss = torch.nn.MSELoss()(values, is_ratios * targets) / 2
        return value_loss, {"loss(critic)": value_loss.item()}

    def sampled_one_step_state_values(self, next_obs, rewards, dones):
        """Bootstrapped approximation of true state-value using sampled transition."""
        return torch.where(
            dones,
            rewards,
            rewards + self.config["gamma"] * self.target_critic(next_obs).squeeze(-1),
        )


class ISSoftVIteration:
    """Loss function for Importance Sampled Soft V Iteration.

    Args:
        critic (callable): state-value function
        target_critic (callable): state-value function for the next state
        alpha (callable): entropy coefficient schedule
        gamma (float): discount factor
    """

    IS_RATIOS = "is_ratios"
    ENTROPY = "entropy"

    def __init__(self, critic, target_critic, alpha, **config):
        self.critic = critic
        self.target_critic = target_critic
        self.alpha = alpha
        self.config = config

    def __call__(self, batch):
        """Compute loss for importance sampled soft V iteration."""
        obs, next_obs, rewards, dones, is_ratios, entropy = dutil.get_keys(
            batch,
            SampleBatch.CUR_OBS,
            SampleBatch.NEXT_OBS,
            SampleBatch.REWARDS,
            SampleBatch.DONES,
            self.IS_RATIOS,
            self.ENTROPY,
        )

        values = self.critic(obs).squeeze(-1)
        with torch.no_grad():
            targets = self.sampled_one_step_state_values(
                entropy, next_obs, rewards, dones
            )
        value_loss = torch.nn.MSELoss()(values, is_ratios * targets) / 2
        return value_loss, {"loss(critic)": value_loss.item()}

    def sampled_one_step_state_values(self, entropy, next_obs, rewards, dones):
        """Bootstrapped approximation of true state-value using sampled transition."""
        gamma = self.config["gamma"]
        augmented_rewards = rewards + self.alpha() * entropy
        return torch.where(
            dones,
            augmented_rewards,
            augmented_rewards + gamma * self.target_critic(next_obs).squeeze(-1),
        )
