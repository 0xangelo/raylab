"""Losses for computing policy gradients."""
from typing import Tuple

import torch
import torch.nn as nn
from ray.rllib import SampleBatch

from raylab.policy.modules.v0.mixins.stochastic_actor_mixin import StochasticPolicy
from raylab.utils.annotations import DetPolicy

from .abstract import Loss


class DeterministicPolicyGradient(Loss):
    """Loss function for Deterministic Policy Gradient.

    Args:
        actor: deterministic policy
        critics: callables for action-values
    """

    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)

    def __init__(self, actor: DetPolicy, critics: nn.ModuleList):
        self.actor = actor
        self.critics = critics

    def __call__(self, batch):
        obs = batch[SampleBatch.CUR_OBS]

        values = self.state_value(obs)
        loss = -torch.mean(values)

        stats = {"loss(actor)": loss.item()}
        return loss, stats

    def state_value(self, obs):
        """Compute the state value by combining policy and action-value function."""
        actions = self.actor(obs)
        return self.critics(obs, actions, clip=True)[..., 0]


class ReparameterizedSoftPG(Loss):
    """Loss function for Soft Policy Iteration with reparameterized actor.

    Args:
        actor: stochastic reparameterized policy
        critics: callables for action-values

    Attributes:
        alpha: entropy coefficient schedule
    """

    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)
    alpha: float = 0.05

    def __init__(
        self, actor: StochasticPolicy, critics: nn.ModuleList,
    ):
        self.actor = actor
        self.critics = critics

    def __call__(self, batch):
        obs = batch[SampleBatch.CUR_OBS]

        action_values, entropy = self.action_value_plus_entropy(obs)
        loss = -torch.mean(action_values + self.alpha * entropy)

        stats = {"loss(actor)": loss.item(), "entropy": entropy.mean().item()}
        return loss, stats

    def action_value_plus_entropy(self, obs):
        """
        Compute the action-value and a single sample estimate of the policy's entropy.
        """
        actions, logp = self.actor.rsample(obs)
        action_values = self.critics(obs, actions, clip=True)[..., 0]
        return action_values, -logp
