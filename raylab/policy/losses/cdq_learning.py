"""Modularized Q-Learning procedures."""
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from torch import Tensor

import raylab.utils.dictionaries as dutil
from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy
from raylab.policy.modules.actor.policy.stochastic import StochasticPolicy
from raylab.policy.modules.critic.q_value import QValueEnsemble
from raylab.policy.modules.critic.v_value import PolicyQValue
from raylab.policy.modules.critic.v_value import VValueEnsemble
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict

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
                f"Q{i}_max": q.max().item(),
                f"Q{i}_min": q.min().item(),
            }
            info.update(infoi)
        return info


class ClippedDoubleQLearning(QLearningMixin, Loss):
    """Clipped Double Q-Learning.

    Use the minimun of two target value functions as the next state-value in
    the target for fitted Q iteration.

    Args:
        critics: Main action-values
        target_critics: Target value functions. If provided an action-value
            function, requires a deterministic policy as the `actor` argument.
        actor: Optional deterministic policy for the next state

    Attributes:
        gamma: discount factor
    """

    gamma: float = 0.99

    def __init__(
        self,
        critics: QValueEnsemble,
        target_critics: Union[QValueEnsemble, VValueEnsemble],
        actor: Optional[DeterministicPolicy] = None,
    ):
        self.critics = critics
        if isinstance(target_critics, QValueEnsemble):
            if actor is None:
                raise ValueError(
                    f"Passing a Q-value function to {type(self).__name__}"
                    " requires a deterministic policy as the `actor` argument."
                )
            self.target_critics = VValueEnsemble(
                [PolicyQValue(policy=actor, q_value=q) for q in target_critics]
            )
        else:
            self.target_critics = target_critics

    def critic_targets(self, rewards, next_obs, dones):
        clipped = VValueEnsemble.clipped(self.target_critics(next_obs))
        next_vals = torch.where(dones, torch.zeros_like(clipped), clipped)
        return rewards + self.gamma * next_vals


class SoftCDQLearning(QLearningMixin, Loss):
    """Clipped Double Q-Learning for maximum entropy RL.

    Args:
        critics: Main action-values
        target_critics: Target action-values
        actor: Stochastic policy for the next state

    Attributes:
        gamma: discount factor
        alpha: entropy coefficient
    """

    gamma: float = 0.99
    alpha: float = 0.05

    def __init__(
        self,
        critics: QValueEnsemble,
        target_critics: QValueEnsemble,
        actor: StochasticPolicy,
    ):
        self.critics = critics
        self.target_critics = target_critics
        self.actor = actor

    def critic_targets(self, rewards, next_obs, dones):
        next_acts, next_logp = self.actor.sample(next_obs)
        clipped = QValueEnsemble.clipped(self.target_critics(next_obs, next_acts))

        next_vals = torch.where(dones, torch.zeros_like(clipped), clipped)
        next_entropy = torch.where(dones, torch.zeros_like(next_logp), -next_logp)
        return rewards + self.gamma * (next_vals + self.alpha * next_entropy)
