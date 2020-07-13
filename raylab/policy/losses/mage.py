"""Losses for model aware action gradient estimation."""
from dataclasses import dataclass
from typing import Dict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy
from raylab.policy.modules.critic.q_value import QValueEnsemble
from raylab.policy.modules.model.stochastic.ensemble import StochasticModelEnsemble

from .abstract import Loss
from .mixins import EnvFunctionsMixin
from .mixins import UniformModelPriorMixin


@dataclass
class MAGEModules:
    """Necessary modules for MAGE loss.

    Attributes:
        critics: Q-value estimators
        target_critics: Q-value estimators for the next state
        policy: deterministic policy for current state
        target_policy: deterministic policy for next state
        models: ensemble of stochastic models
    """

    critics: QValueEnsemble
    target_critics: QValueEnsemble
    policy: DeterministicPolicy
    target_policy: DeterministicPolicy
    models: StochasticModelEnsemble


class MAGE(EnvFunctionsMixin, UniformModelPriorMixin, Loss):
    """Loss function for Model-based Action-Gradient-Estimator.

    Args:
        modules: necessary modules for MAGE loss

    Attributes:
        gamma: discount factor
        lambda: weighting factor for TD-error regularization
    """

    batch_keys = (SampleBatch.CUR_OBS,)
    gamma: float = 0.99
    lambda_: float = 0.05

    def __init__(self, modules: MAGEModules):
        super().__init__()
        self._modules = nn.ModuleDict(
            dict(
                critics=modules.critics,
                target_critics=modules.target_critics,
                policy=modules.policy,
                target_policy=modules.target_policy,
                models=modules.models,
            )
        )
        self._rng = np.random.default_rng()

    @property
    def initialized(self) -> bool:
        """Whether or not the loss function has all the necessary components."""
        return self._env.initialized

    @property
    def grad_estimator(self):
        """Gradient estimator for expecations."""
        return "PD"

    @property
    def model_samples(self):
        """Number of next states to draw from the model"""
        return 1

    @property
    def _models(self) -> StochasticModelEnsemble:
        return self._modules["models"]

    def compile(self):
        self._modules.update({k: torch.jit.script(v) for k, v in self._modules.items()})

    def transition(self, obs, action):
        next_obs, _ = super().transition(obs, action)
        return next_obs.squeeze(dim=0)

    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        assert self.initialized, (
            "Environment functions missing. "
            "Did you set reward, termination, and dynamics functions?"
        )

        obs = batch[SampleBatch.CUR_OBS]
        action = self._modules["policy"](obs)
        next_obs = self.transition(obs, action)

        delta = self.temporal_diff_error(obs, action, next_obs)
        grad_loss = self.gradient_loss(delta, action)
        td_reg = self.temporal_diff_loss(delta)
        loss = grad_loss + self.lambda_ * td_reg

        info = {
            "loss(critics)": loss.item(),
            "loss(MAGE)": grad_loss.item(),
            "loss(TD)": td_reg.item(),
        }
        return loss, info

    def temporal_diff_error(
        self, obs: Tensor, action: Tensor, next_obs: Tensor,
    ) -> Tensor:
        """Returns the temporal difference error with clipped action values."""
        values = torch.cat([m(obs, action) for m in self._modules["critics"]], dim=-1)

        reward = self._env.reward(obs, action, next_obs)
        done = self._env.termination(obs, action, next_obs)
        next_action = self._modules["target_policy"](next_obs)
        next_values, _ = torch.cat(
            [m(next_obs, next_action) for m in self._modules["target_critics"]], dim=-1
        ).min(dim=-1)
        targets = torch.where(done, reward, reward + self.gamma * next_values)
        targets = targets.unsqueeze(-1).expand_as(values)

        return targets - values

    @staticmethod
    def gradient_loss(delta: Tensor, action: Tensor) -> Tensor:
        """Returns the action gradient loss for the Q-value function."""
        (action_gradient,) = torch.autograd.grad(delta.sum(), action, create_graph=True)
        return action_gradient.abs().sum(dim=-1).mean()

    @staticmethod
    def temporal_diff_loss(delta: Tensor) -> Tensor:
        """Returns the temporal difference loss for the Q-value function."""
        return delta.abs().sum(dim=-1).mean()
