"""Losses for Importance Sampled Fitted V Iteration."""
from typing import Tuple

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils import override
from torch import Tensor

import raylab.utils.dictionaries as dutil
from raylab.policy.modules.actor.policy.stochastic import StochasticPolicy
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict

from .abstract import Loss


class ISFittedVIteration(Loss):
    """Loss function for Importance Sampled Fitted V Iteration.

    Args:
        critic: state-value function
        target_critic: state-value function for the next state

    Attributes:
        gamma: discount factor
    """

    IS_RATIOS = "is_ratios"
    batch_keys: Tuple[str, str] = (SampleBatch.CUR_OBS, "is_ratios")
    gamma: float = 0.99

    def __init__(self, critic: nn.Module, target_critic: nn.Module):
        self.critic = critic
        self.target_critic = target_critic

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        """Compute loss for importance sampled fitted V iteration."""
        obs, is_ratios = dutil.get_keys(batch, *self.batch_keys)

        values = self.critic(obs).squeeze(-1)
        with torch.no_grad():
            targets = self.sampled_one_step_state_values(batch)
        value_loss = torch.mean(
            is_ratios * torch.nn.MSELoss(reduction="none")(values, targets) / 2
        )
        return value_loss, {"loss(critic)": value_loss.item()}

    def sampled_one_step_state_values(self, batch: TensorDict) -> Tensor:
        """Bootstrapped approximation of true state-value using sampled transition."""
        next_obs, rewards, dones = dutil.get_keys(
            batch, SampleBatch.NEXT_OBS, SampleBatch.REWARDS, SampleBatch.DONES,
        )
        return torch.where(
            dones,
            rewards,
            rewards + self.gamma * self.target_critic(next_obs).squeeze(-1),
        )


class ISSoftVIteration(ISFittedVIteration):
    """Loss function for Importance Sampled Soft V Iteration.

    Args:
        critic: state-value function
        target_critic: state-value function for the next state
        actor: stochastic policy

    Attributes:
        gamma: discount factor
        alpha: entropy coefficient
    """

    # pylint:disable=too-few-public-methods
    ENTROPY = "entropy"
    gamma: float = 0.99
    alpha: float = 0.05

    def __init__(
        self, critic: nn.Module, target_critic: nn.Module, actor: StochasticPolicy
    ):
        super().__init__(critic, target_critic)
        self.actor = actor

    @override(ISFittedVIteration)
    def sampled_one_step_state_values(self, batch: TensorDict) -> Tensor:
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
        augmented_rewards = rewards + self.alpha * entropy
        return torch.where(
            dones,
            augmented_rewards,
            augmented_rewards + self.gamma * self.target_critic(next_obs).squeeze(-1),
        )
