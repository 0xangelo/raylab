"""Modularized Q-Learning procedures."""
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

import raylab.utils.dictionaries as dutil
from raylab.utils.annotations import DetPolicy
from raylab.utils.annotations import StochasticPolicy

from .abstract import Loss
from .utils import clipped_action_value


class QLearningMixin:
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
        values = torch.cat([m(obs, actions) for m in self.critics], dim=-1)
        critic_loss = loss_fn(values, target_values.unsqueeze(-1).expand_as(values))

        stats = {
            "q_mean": values.mean().item(),
            "q_max": values.max().item(),
            "q_min": values.min().item(),
            "loss(critics)": critic_loss.item(),
        }
        return critic_loss, stats


class ClippedDoubleQLearning(QLearningMixin, Loss):
    """Clipped Double Q-Learning.

    Use the minimun of two target Q functions as the next action-value in the target
    for fitted Q iteration.

    Args:
        critics: callables for main action-values
        target_critics: callables for target action-values
        actor: deterministic policy for the next state

    Attributes:
        gamma: discount factor
    """

    gamma: float = 0.99

    def __init__(
        self, critics: nn.ModuleList, target_critics: nn.ModuleList, actor: DetPolicy,
    ):
        self.critics = critics
        self.target_critics = target_critics
        self.actor = actor

    def critic_targets(self, rewards, next_obs, dones):
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Clipped Double Q-Learning
        using target networks and batch transitions.
        """
        next_acts = self.actor(next_obs)
        target_values = clipped_action_value(next_obs, next_acts, self.target_critics)
        next_values = torch.where(dones, torch.zeros_like(target_values), target_values)
        return rewards + self.gamma * next_values


class SoftCDQLearning(QLearningMixin, Loss):
    """Clipped Double Q-Learning for maximum entropy RL.

    Args:
        critics: callables for main action-values
        target_critics: callables for target action-values
        actor: stochastic policy for the next state

    Attributes:
        gamma: discount factor
        alpha: entropy coefficient
    """

    gamma: float = 0.99
    alpha: float = 0.05

    def __init__(
        self,
        critics: nn.ModuleList,
        target_critics: nn.ModuleList,
        actor: StochasticPolicy,
    ):
        self.critics = critics
        self.target_critics = target_critics
        self.actor = actor

    def critic_targets(self, rewards, next_obs, dones):
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Clipped Double Q-Learning
        using target networks and batch transitions.
        """
        next_acts, next_logp = self.actor(next_obs)
        target_values = clipped_action_value(next_obs, next_acts, self.target_critics)

        next_values = torch.where(dones, torch.zeros_like(target_values), target_values)
        next_entropy = torch.where(dones, torch.zeros_like(next_logp), -next_logp)
        return rewards + self.gamma * (next_values + self.alpha * next_entropy)
