"""Modularized Q-Learning procedures."""
from abc import ABC
from abc import abstractmethod
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
from raylab.policy.modules.critic.q_value import QValue
from raylab.policy.modules.critic.v_value import PolicyQValue
from raylab.policy.modules.critic.v_value import VValue
from raylab.policy.modules.model.stochastic.ensemble import SME
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict

from .abstract import Loss
from .mixins import EnvFunctionsMixin
from .mixins import UniformModelPriorMixin
from .utils import dist_params_stats


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

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, TensorDict]:
        """Compute loss for Q-value function."""
        obs, actions, rewards, next_obs, dones = dutil.get_keys(batch, *self.batch_keys)
        with torch.no_grad():
            target_values = self.critic_targets(rewards, next_obs, dones)
        loss_fn = nn.MSELoss()
        values = self.critics(obs, actions)
        critic_loss = loss_fn(values, target_values)

        stats = {"loss(critics)": critic_loss.item()}
        stats.update(self.q_value_info(values))
        return critic_loss, stats

    @abstractmethod
    def critic_targets(
        self, rewards: Tensor, next_obs: Tensor, dones: Tensor
    ) -> Tensor:
        """Compute clipped 1-step approximation of Q^{\\pi}(s, a)."""

    @staticmethod
    def q_value_info(value: Tensor) -> StatDict:
        """Return the average, min, and max Q-values in a batch."""
        info = {
            "q_mean": value.mean().item(),
            "q_max": value.max().item(),
            "q_min": value.min().item(),
        }
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
        critics: QValue,
        target_critics: Union[QValue, VValue],
        actor: Optional[DeterministicPolicy] = None,
    ):
        self.critics = critics
        if isinstance(target_critics, QValue):
            if actor is None:
                raise ValueError(
                    f"Passing a Q-value function to {type(self).__name__}"
                    " requires a deterministic policy as the `actor` argument."
                )
            self.target_critics = PolicyQValue(policy=actor, q_value=target_critics)
        else:
            self.target_critics = target_critics

    def critic_targets(self, rewards, next_obs, dones):
        unclipped_values = self.target_critics(next_obs)
        values, _ = unclipped_values.min(dim=-1)
        next_vals = torch.where(dones, torch.zeros_like(values), values)
        target = rewards + self.gamma * next_vals
        return target.unsqueeze(-1).expand_as(unclipped_values)


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
        self, critics: QValue, target_critics: QValue, actor: StochasticPolicy,
    ):
        self.critics = critics
        self.target_critics = target_critics
        self.actor = actor

    def critic_targets(self, rewards, next_obs, dones):
        next_acts, next_logp = self.actor.sample(next_obs)
        unclipped_values = self.target_critics(next_obs, next_acts)
        values, _ = unclipped_values.min(dim=-1)

        next_vals = torch.where(dones, torch.zeros_like(values), values)
        next_entropy = torch.where(dones, torch.zeros_like(next_logp), -next_logp)
        target = rewards + self.gamma * (next_vals + self.alpha * next_entropy)
        return target.unsqueeze(-1).expand_as(unclipped_values)


class DynaSoftCDQLearning(EnvFunctionsMixin, UniformModelPriorMixin, SoftCDQLearning):
    """Loss function Dyna-augmented soft clipped double Q-learning.

    Args:
        critics: Main action-values
        models: Stochastic model ensemble
        target_critics: Target action-values
        actor: Stochastic policy for the next state

    Attributes:
        gamma: discount factor
        alpha: entropy coefficient
    """

    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)

    def __init__(
        self,
        critics: QValue,
        models: SME,
        target_critics: QValue,
        actor: StochasticPolicy,
    ):
        super().__init__(critics, target_critics, actor)
        self._models = models

    @property
    def initialized(self) -> bool:
        """Whether or not the loss function has all the necessary components."""
        return self._env.initialized

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        assert self.initialized, (
            "Environment functions missing. "
            "Did you set reward and termination functions?"
        )
        obs = batch[SampleBatch.CUR_OBS]
        action, _ = self.actor.sample(obs)
        next_obs, _, dist_params = self.transition(obs, action)

        reward = self._env.reward(obs, action, next_obs)
        done = self._env.termination(obs, action, next_obs)

        loss_fn = nn.MSELoss()
        value = self.critics(obs, action)
        with torch.no_grad():
            target_value = self.critic_targets(reward, next_obs, done)
        loss = loss_fn(value, target_value)

        stats = {"loss(critics)": loss.item()}
        stats.update(self.q_value_info(value))
        stats.update(dist_params_stats(dist_params, name="model"))
        return loss, stats
