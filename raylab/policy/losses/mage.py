"""Losses for model aware action gradient estimation."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy
from raylab.policy.modules.critic.q_value import QValueEnsemble
from raylab.policy.modules.model.stochastic.ensemble import SME
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict

from .abstract import Loss
from .mixins import EnvFunctionsMixin
from .mixins import UniformModelPriorMixin
from .utils import dist_params_stats


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
    models: SME


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
    def _models(self) -> SME:
        return self._modules["models"]

    def transition(self, obs, action):
        next_obs, _, dist_params = super().transition(obs, action)
        return next_obs, dist_params

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        assert self.initialized, (
            "Environment functions missing. "
            "Did you set reward, termination, and dynamics functions?"
        )

        obs = batch[SampleBatch.CUR_OBS]
        action = self._modules["policy"](obs)
        next_obs, dist_params = self.transition(obs, action)

        delta = self.temporal_diff_error(obs, action, next_obs)
        grad_loss = self.gradient_loss(delta, action)
        td_reg = self.temporal_diff_loss(delta)
        loss = grad_loss + self.lambda_ * td_reg

        info = {
            "loss(critics)": loss.item(),
            "loss(MAGE)": grad_loss.item(),
            "loss(TD)": td_reg.item(),
        }
        info.update(dist_params_stats(dist_params, name="model"))
        return loss, info

    def temporal_diff_error(
        self, obs: Tensor, action: Tensor, next_obs: Tensor,
    ) -> Tensor:
        """Returns the temporal difference error with clipped action values."""
        critics = self._modules["critics"]
        target_policy = self._modules["target_policy"]
        target_critics = self._modules["target_critics"]

        values = critics(obs, action)  # (*, N)
        reward = self._env.reward(obs, action, next_obs)  # (*,)
        done = self._env.termination(obs, action, next_obs)  # (*,)
        next_action = target_policy(next_obs)  # (*, A)
        next_values, _ = target_critics(next_obs, next_action).min(dim=-1)  # (*,)
        targets = torch.where(done, reward, reward + self.gamma * next_values)  # (*,)
        targets = targets.unsqueeze(-1).expand_as(values)  # (*, N)

        return targets - values

    @staticmethod
    def gradient_loss(delta: Tensor, action: Tensor) -> Tensor:
        """Returns the action gradient loss for the Q-value function."""
        (action_gradient,) = torch.autograd.grad(delta.sum(), action, create_graph=True)
        return torch.sum(action_gradient ** 2, dim=-1).mean()

    @staticmethod
    def temporal_diff_loss(delta: Tensor) -> Tensor:
        """Returns the temporal difference loss for the Q-value function."""
        return torch.sum(delta ** 2, dim=-1).mean()
