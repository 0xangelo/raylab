"""Loss functions for Decision-Aware Model Learning."""
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.utils.annotations import ActionValue
from raylab.utils.annotations import DetPolicy
from raylab.utils.annotations import DynamicsFn
from raylab.utils.annotations import RewardFn

from .abstract import Loss


class DPGAwareModelLearning(Loss):
    """Loss function for Deterministic Policy Gradient-Aware model learning.

    Args:
        actor: deterministic policy
        critics: callables for action-values

    Atributes:
        gamma: discount factor
        grad_estimator: gradient estimator for expectations. One of 'PD' or 'SF'
        model: stochastic model that returns next state and its log density
        reward_fn: callable that computes rewards for transitions
    """

    gamma: float = 0.99
    grad_estimator: str = "SF"
    model: Optional[DynamicsFn]
    reward_fn: Optional[RewardFn]
    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)

    def __init__(
        self, actor: DetPolicy, critics: List[ActionValue],
    ):
        self.actor = actor
        self.critics = critics

        self.model = None
        self.reward_fn = None

    def set_model(self, function: DynamicsFn):
        """Set model to provided callable."""
        self.model = function

    def set_reward_fn(self, function: RewardFn):
        """Set reward function to provided callable."""
        self.reward_fn = function

    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, dict]:
        """Compute policy gradient-aware (PGA) model loss."""
        obs = batch[SampleBatch.CUR_OBS]
        actions = self.actor(obs).detach().requires_grad_()

        predictions = self.one_step_action_value_surrogate(obs, actions)
        targets = self.zero_step_action_values(obs, actions)

        temporal_diff = torch.sum(targets - predictions)
        (action_gradients,) = torch.autograd.grad(
            temporal_diff, actions, create_graph=True
        )

        daml_loss = torch.sum(action_gradients * action_gradients, dim=-1).mean()
        return (
            daml_loss,
            {"loss(action)": temporal_diff.item(), "loss(model)": daml_loss.item()},
        )

    def one_step_action_value_surrogate(
        self, obs: Tensor, actions: Tensor, model_samples: int = 1
    ) -> Tensor:
        """
        Compute 1-step approximation of Q^{\\pi}(s, a) for Deterministic Policy Gradient
        using target networks and model transitions.
        """
        next_obs, rewards, logp = self._generate_transition(obs, actions, model_samples)
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

    def _generate_transition(
        self, obs: Tensor, actions: Tensor, num_samples: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute virtual transition and its log density."""
        sample_shape = (num_samples,)
        obs = obs.expand(sample_shape + obs.shape)
        actions = actions.expand(sample_shape + actions.shape)

        next_obs, logp = self.model(obs, actions)
        rewards = self.reward_fn(obs, actions, next_obs)
        return next_obs, rewards, logp

    def zero_step_action_values(self, obs: Tensor, actions: Tensor) -> Tensor:
        """Compute Q^{\\pi}(s, a) directly using approximate critic."""
        return self.critics(obs, actions, clip=True)[..., 0]
