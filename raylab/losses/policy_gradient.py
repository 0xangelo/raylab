"""Losses for computing policy gradients."""
from typing import Dict
from typing import List
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.utils.annotations import ActionValue
from raylab.utils.annotations import DetPolicy
from raylab.utils.annotations import DynamicsFn
from raylab.utils.annotations import RewardFn

from .utils import clipped_action_value


class DeterministicPolicyGradient:
    """Loss function for Deterministic Policy Gradient.

    Args:
        actor (callable): deterministic policy
        critics (list): callables for action-values
    """

    def __init__(self, actor, critics):
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
        return clipped_action_value(obs, actions, self.critics)


class ReparameterizedSoftPG:
    """Loss function for Soft Policy Iteration with reparameterized actor.

    Args:
        actor (StochasticPolicy): stochastic reparameterized policy
        critics (list): callables for action-values
        alpha (callable): entropy coefficient schedule
        rlogp (bool): whether to draw reparameterized log_probs from the actor
    """

    def __init__(self, actor, critics, alpha):
        self.actor = actor
        self.critics = critics
        self.alpha = alpha

    def __call__(self, batch):
        obs = batch[SampleBatch.CUR_OBS]

        action_values, entropy = self.action_value_entropy(obs)
        loss = -torch.mean(action_values + self.alpha() * entropy)

        stats = {"loss(actor)": loss.item(), "entropy": entropy.mean().item()}
        return loss, stats

    def action_value_entropy(self, obs):
        """
        Compute the action-value and a single sample estimate of the policy's entropy.
        """
        actions, logp = self.actor.rsample(obs)
        action_values = clipped_action_value(obs, actions, self.critics)
        return action_values, -logp


class ModelAwareDPG:
    """Loss function for Model-Aware Deterministic Policy Gradient.

    Args:
        actor: deterministic policy
        critics: callables for action-values
        gamma (float): discount factor
        num_model_samples (int): number of next states to draw from the model
        grad_estimator (str): gradient estimator for expecations ('PD' or 'SF')

    Attributes:
        reward_fn (Optional[RewardFn]): reward function for state, action, and
            next state tuples
        model (Optional[DynamicsFn]): stochastic model that returns next state
            and its log density
    """

    def __init__(self, actor: DetPolicy, critics: List[ActionValue], **config):
        self.actor = actor
        self.critics = critics
        self.config = config

        self.reward_fn = None
        self.model = None

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
            obs, actions, self.config["num_model_samples"]
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
        next_values = clipped_action_value(next_obs, next_acts, self.critics)
        values = rewards + self.config["gamma"] * next_values

        if self.config["grad_estimator"] == "SF":
            surrogate = torch.mean(logp * values.detach(), dim=0)
        elif self.config["grad_estimator"] == "PD":
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
