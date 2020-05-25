"""Modularized Q-Learning procedures."""
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from ray.rllib.utils import override

import raylab.utils.dictionaries as dutil

from .utils import clipped_action_value


class ClippedDoubleQLearning:
    """Clipped Double Q-Learning.

    Use the minimun of two target Q functions as the next action-value in the target
    for fitted Q iteration.

    Args:
        critics (list): callables for main action-values
        target_critics (list): callables for target action-values
        actor (callable): deterministic policy for the next state
        gamma (float): discount factor
    """

    def __init__(self, critics, target_critics, actor, gamma):
        self.critics = critics
        self.target_critics = target_critics
        self.actor = actor
        self.gamma = gamma

    def __call__(self, batch):
        """Compute loss for Q-value function."""
        # pylint:disable=too-many-arguments
        obs, actions, rewards, next_obs, dones = dutil.get_keys(
            batch,
            SampleBatch.CUR_OBS,
            SampleBatch.ACTIONS,
            SampleBatch.REWARDS,
            SampleBatch.NEXT_OBS,
            SampleBatch.DONES,
        )
        with torch.no_grad():
            target_values = self.critic_targets(rewards, next_obs, dones)
        loss_fn = nn.MSELoss()
        values = torch.cat([m(obs, actions) for m in self.critics], dim=-1)
        critic_loss = loss_fn(values, target_values.unsqueeze(-1).expand_as(values))

        stats = {
            "q_mean": values.mean().item(),
            "q_max": values.max().item(),
            "q_min": values.min().item(),
            "loss(critic)": critic_loss.item(),
        }
        return critic_loss, stats

    def critic_targets(self, rewards, next_obs, dones):
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Clipped Double Q-Learning
        using target networks and batch transitions.
        """
        next_acts = self.actor(next_obs)
        target_values = clipped_action_value(next_obs, next_acts, self.target_critics)
        next_values = torch.where(dones, torch.zeros_like(target_values), target_values)
        return rewards + self.gamma * next_values


class SoftCDQLearning(ClippedDoubleQLearning):
    """Clipped Double Q-Learning for maximum entropy RL.

    Args:
        critics (list): callables for main action-values
        target_critics (list): callables for target action-values
        actor (callable): stochastic policy for the next state
        gamma (float): discount factor
        alpha (callable): entropy coefficient schedule
    """

    # pylint:disable=too-few-public-methods

    def __init__(self, critics, target_critics, actor, gamma, alpha):
        # pylint:disable=too-many-arguments
        super().__init__(critics, target_critics, actor.sample, gamma)
        self.alpha = alpha

    @override(ClippedDoubleQLearning)
    def critic_targets(self, rewards, next_obs, dones):
        next_acts, next_logp = self.actor(next_obs)
        target_values = clipped_action_value(next_obs, next_acts, self.target_critics)

        next_values = torch.where(dones, torch.zeros_like(target_values), target_values)
        next_entropy = torch.where(dones, torch.zeros_like(next_logp), -next_logp)
        return rewards + self.gamma * (next_values + self.alpha() * next_entropy)
