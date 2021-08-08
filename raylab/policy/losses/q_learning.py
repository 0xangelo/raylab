"""Modularized Q-Learning procedures."""
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from nnrl.nn.critic import QValueEnsemble, VValue
from nnrl.types import TensorDict
from ray.rllib import SampleBatch
from torch import Tensor, nn

import raylab.utils.dictionaries as dutil
from raylab.utils.types import StatDict

from .abstract import Loss


class QLearningMixin(ABC):
    """Adds default call for Q-Learning losses."""

    # pylint:disable=too-few-public-methods
    batch_keys = (
        SampleBatch.CUR_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.REWARDS,
        SampleBatch.NEXT_OBS,
        SampleBatch.DONES,
    )
    critics: QValueEnsemble

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, TensorDict]:
        """Compute loss for Q-value function."""
        obs, actions, rewards, next_obs, dones = dutil.get_keys(batch, *self.batch_keys)
        with torch.no_grad():
            target_values = self.critic_targets(rewards, next_obs, dones)
        loss_fn = nn.MSELoss()
        values = self.critics(obs, actions)
        critic_loss = torch.stack([loss_fn(v, target_values) for v in values]).sum()

        stats = {"loss(critics)": critic_loss.item()}
        stats.update(self.q_value_info(values))
        return critic_loss, stats

    @abstractmethod
    def critic_targets(
        self, rewards: Tensor, next_obs: Tensor, dones: Tensor
    ) -> Tensor:
        """Compute clipped 1-step approximation of Q^{\\pi}(s, a)."""

    @staticmethod
    def q_value_info(values: List[Tensor]) -> StatDict:
        """Return the average, min, and max Q-values in a batch."""
        info = {}
        # pylint:disable=invalid-name
        for i, q in enumerate(values):
            infoi = {
                f"Q{i}_mean": q.mean().item(),
                f"Q{i}_std": q.std().item(),
                f"Q{i}_max": q.max().item(),
                f"Q{i}_min": q.min().item(),
            }
            info.update(infoi)
        return info


class FittedQLearning(QLearningMixin, Loss):
    """Fitted Q-Learning.

    Generalized fitted Q-Learning implementation using a state-value function as
    target.

    Clipped Double Q-Learning can be achieved by composing the two target
    critics into a state-value function using, e.g., ClippedQValue and HardValue

    Args:
        critics: Main action-values
        target_critic: Target state-value function.

    Attributes:
        gamma: discount factor
    """

    gamma: float = 0.99

    def __init__(
        self,
        critics: QValueEnsemble,
        target_critic: VValue,
    ):
        self.critics = critics
        assert isinstance(
            target_critic, VValue
        ), "Need state-value function for critic target."
        self.target_critic = target_critic

    def critic_targets(self, rewards, next_obs, dones):
        values = self.target_critic(next_obs)
        values = torch.where(dones, torch.zeros_like(values), values)
        return rewards + self.gamma * values
