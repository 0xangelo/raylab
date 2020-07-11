"""Losses for computing policy gradients."""
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.policy.modules.v0.mixins.stochastic_actor_mixin import StochasticPolicy
from raylab.utils.annotations import DetPolicy
from raylab.utils.annotations import DynamicsFn
from raylab.utils.annotations import RewardFn

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


class ModelAwareDPG:
    """Loss function for Model-Aware Deterministic Policy Gradient.

    Args:
        actor: deterministic policy
        critics: callables for action-values

    Attributes:
        gamma: discount factor
        num_model_samples: number of next states to draw from the model
        grad_estimator: gradient estimator for expecations ('PD' or 'SF')
        reward_fn: reward function for state, action, and
            next state tuples
        model: stochastic model that returns next state
            and its log density
    """

    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)
    gamma: float = 0.99
    num_model_samples: int = 1
    grad_estimator: str = "SF"
    reward_fn: Optional[RewardFn] = None
    model: Optional[DynamicsFn] = None

    def __init__(
        self, actor: DetPolicy, critics: nn.ModuleList,
    ):
        self.actor = actor
        self.critics = critics

    def set_reward_fn(self, function: RewardFn):
        """Set reward function to provided callable."""
        self.reward_fn = function

    def set_model(self, function: DynamicsFn):
        """Set model to provided callable."""
        self.model = function

    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        """Compute loss for Model-Aware Deterministic Policy Gradient."""
        obs = batch[SampleBatch.CUR_OBS]

        actions = self.actor(obs)
        action_values = self.one_step_action_value_surrogate(
            obs, actions, self.num_model_samples
        )
        loss = -torch.mean(action_values)

        stats = {"loss(actor)": loss.item()}
        return loss, stats

    def one_step_action_value_surrogate(self, obs, actions, num_samples=1):
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Deterministic Policy Gradient
        using target networks and model transitions.
        """
        next_obs, rewards, logp = self._generate_transition(obs, actions, num_samples)
        # Next action grads shouldn't propagate
        with torch.no_grad():
            next_acts = self.actor(next_obs)
        next_values = self.critics(next_obs, next_acts, clip=True)[..., 0]
        values = rewards + self.gamma * next_values

        if self.grad_estimator == "SF":
            surrogate = torch.mean(logp * values.detach(), dim=0)
        elif self.grad_estimator == "PD":
            surrogate = torch.mean(values, dim=0)
        return surrogate

    def _generate_transition(self, obs, actions, num_samples):
        """Compute virtual transition and its log density."""
        sample_shape = (num_samples,)
        obs = obs.expand(sample_shape + obs.shape)
        actions = actions.expand(sample_shape + actions.shape)

        next_obs, logp = self.model(obs, actions)
        rewards = self.reward_fn(obs, actions, next_obs)
        return next_obs, rewards, logp
