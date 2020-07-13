"""Modularized Q-Learning procedures."""
from abc import ABC
from abc import abstractmethod
from functools import partial
from typing import Dict
from typing import Tuple

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from torch import Tensor

import raylab.utils.dictionaries as dutil
from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy
from raylab.policy.modules.actor.policy.stochastic import StochasticPolicy
from raylab.policy.modules.critic.q_value import QValueEnsemble
from raylab.policy.modules.model.stochastic.ensemble import StochasticModelEnsemble

from .abstract import Loss
from .mixins import EnvFunctionsMixin
from .mixins import UniformModelPriorMixin


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

    def __call__(self, batch):
        """Compute loss for Q-value function."""
        obs, actions, rewards, next_obs, dones = dutil.get_keys(batch, *self.batch_keys)
        with torch.no_grad():
            target_values = self.critic_targets(rewards, next_obs, dones)
        loss_fn = nn.MSELoss()
        values = self.critics(obs, actions)
        critic_loss = loss_fn(values, target_values)

        stats = {
            "q_mean": values.mean().item(),
            "q_max": values.max().item(),
            "q_min": values.min().item(),
            "loss(critics)": critic_loss.item(),
        }
        return critic_loss, stats

    @abstractmethod
    def critic_targets(
        self, rewards: Tensor, next_obs: Tensor, dones: Tensor
    ) -> Tensor:
        """Compute clipped 1-step approximation of Q^{\\pi}(s, a)."""


class ClippedDoubleQLearning(QLearningMixin, Loss):
    """Clipped Double Q-Learning.

    Use the minimun of two target Q functions as the next action-value in the target
    for fitted Q iteration.

    Args:
        critics: Main action-values
        target_critics: Target action-values
        actor: Deterministic policy for the next state

    Attributes:
        gamma: discount factor
    """

    gamma: float = 0.99

    def __init__(
        self,
        critics: QValueEnsemble,
        target_critics: QValueEnsemble,
        actor: DeterministicPolicy,
    ):
        self.critics = critics
        self.target_critics = target_critics
        self.actor = actor

    def critic_targets(self, rewards, next_obs, dones):
        next_acts = self.actor(next_obs)
        target_values = self.target_critics(next_obs, next_acts, clip=True)
        vals = target_values[..., 0]
        next_vals = torch.where(dones, torch.zeros_like(vals), vals)
        target = rewards + self.gamma * next_vals
        return target.unsqueeze(-1).expand_as(target_values)


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
        target_values = self.target_critics(next_obs, next_acts, clip=True)
        vals = target_values[..., 0]

        next_vals = torch.where(dones, torch.zeros_like(vals), vals)
        next_entropy = torch.where(dones, torch.zeros_like(next_logp), -next_logp)
        target = rewards + self.gamma * (next_vals + self.alpha * next_entropy)
        return target.unsqueeze(-1).expand_as(target_values)


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
        critics: QValueEnsemble,
        models: StochasticModelEnsemble,
        target_critics: QValueEnsemble,
        actor: StochasticPolicy,
    ):
        super().__init__(critics, target_critics, actor)
        self._models = models

    @property
    def initialized(self) -> bool:
        """Whether or not the loss function has all the necessary components."""
        return self._env.initialized

    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        assert self.initialized, (
            "Environment functions missing. "
            "Did you set reward and termination functions?"
        )
        obs = batch[SampleBatch.CUR_OBS]
        action, _ = self.actor.sample(obs)
        next_obs, _ = map(partial(torch.squeeze, dim=0), self.transition(obs, action))
        reward = self._env.reward(obs, action, next_obs)
        done = self._env.termination(obs, action, next_obs)

        loss_fn = nn.MSELoss()
        value = self.critics(obs, action)
        with torch.no_grad():
            target_value = self.critic_targets(reward, next_obs, done)
        critic_loss = loss_fn(value, target_value)

        stats = {
            "q_mean": value.mean().item(),
            "q_max": value.max().item(),
            "q_min": value.min().item(),
            "loss(critics)": critic_loss.item(),
        }
        return critic_loss, stats
