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


class MAGE(EnvFunctionsMixin, Loss):
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

    def compile(self):
        self._modules.update({k: torch.jit.script(v) for k, v in self._modules.items()})

    def seed(self, seed: int):
        """Seeds the RNG for choosing a model from the ensemble."""
        self._rng = np.random.default_rng(seed)

    @property
    def initialized(self) -> bool:
        """Whether or not the loss function has all the necessary components."""
        return self._env.initialized

    def __call__(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
        assert self.initialized, (
            "Environment functions missing. "
            "Did you set reward, termination, and dynamics functions?"
        )

        obs, action, next_obs = self.generate_transition(batch)

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

    def generate_transition(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Sample (s, a, s') tuples given initial states in batch."""
        obs = batch[SampleBatch.CUR_OBS]
        action = self._modules["policy"](obs)

        model = self._rng.choice(list(self._modules["models"]))
        next_obs, _ = model.rsample(obs, action)
        return obs, action, next_obs

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
