"""Modularized Q-Learning procedures."""
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

import raylab.utils.dictionaries as dutil
from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy
from raylab.policy.modules.actor.policy.stochastic import StochasticPolicy
from raylab.policy.modules.critic.q_value import QValueEnsemble

from .abstract import Loss


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
        values = self.critics(obs, actions)
        critic_loss = loss_fn(values, target_values)

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
        self,
        critics: QValueEnsemble,
        target_critics: QValueEnsemble,
        actor: DeterministicPolicy,
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
        target_values = self.target_critics(next_obs, next_acts, clip=True)
        vals = target_values[..., 0]
        next_vals = torch.where(dones, torch.zeros_like(vals), vals)
        target = rewards + self.gamma * next_vals
        return target.unsqueeze(-1).expand_as(target_values)


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
        critics: QValueEnsemble,
        target_critics: QValueEnsemble,
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
        next_acts, next_logp = self.actor.sample(next_obs)
        target_values = self.target_critics(next_obs, next_acts, clip=True)
        vals = target_values[..., 0]

        next_vals = torch.where(dones, torch.zeros_like(vals), vals)
        next_entropy = torch.where(dones, torch.zeros_like(next_logp), -next_logp)
        target = rewards + self.gamma * (next_vals + self.alpha * next_entropy)
        return target.unsqueeze(-1).expand_as(target_values)
