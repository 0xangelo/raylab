"""Losses for model aware action gradient estimation."""
from typing import Tuple, Union

import torch
from nnrl.nn.actor import DeterministicPolicy
from nnrl.nn.critic import QValueEnsemble, VValue
from nnrl.nn.model import SME, StochasticModel
from nnrl.types import TensorDict
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.utils.types import StatDict

from .abstract import Loss
from .mixins import EnvFunctionsMixin, UniformModelPriorMixin
from .utils import dist_params_stats


class MAGE(EnvFunctionsMixin, UniformModelPriorMixin, Loss):
    """Loss function for Model-based Action-Gradient-Estimator.

    Args:
        critics: Q-value estimators
        policy: deterministic policy for current state
        target_critic: V-value estimator for the next state
        models: ensemble of stochastic models

    Attributes:
        gamma: discount factor
        lambd: weighting factor for TD-error regularization
    """

    batch_keys = (SampleBatch.CUR_OBS,)
    gamma: float = 0.99
    lambd: float = 0.05

    def __init__(
        self,
        critics: QValueEnsemble,
        policy: DeterministicPolicy,
        target_critic: VValue,
        models: Union[StochasticModel, SME],
    ):
        super().__init__()
        self.critics = critics
        self.policy = policy
        self.target_critic = target_critic
        if isinstance(models, StochasticModel):
            models = SME([models])
        self.models = models

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        self.check_env_fns()

        obs = batch[SampleBatch.CUR_OBS]
        action = self.policy(obs)
        next_obs, dist_params = self.transition(obs, action)

        delta = self.temporal_diff_error(obs, action, next_obs)
        grad_loss = self.gradient_loss(delta, action)
        td_reg = self.temporal_diff_loss(delta)
        loss = grad_loss + self.lambd * td_reg

        info = {
            "loss(critics)": loss.item(),
            "loss(MAGE)": grad_loss.item(),
            "loss(TD)": td_reg.item(),
        }
        info.update(dist_params_stats(dist_params, name="model"))
        return loss, info

    def transition(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, TensorDict]:
        # pylint:disable=missing-function-docstring
        model, _ = self.sample_model()
        dist_params = model(obs, action)
        next_obs, _ = model.rsample(dist_params)
        return next_obs, dist_params

    def temporal_diff_error(
        self,
        obs: Tensor,
        action: Tensor,
        next_obs: Tensor,
    ) -> Tensor:
        """Returns the temporal difference error."""
        reward = self._env.reward(obs, action, next_obs)  # (*,)
        done = self._env.termination(obs, action, next_obs)  # (*,)
        next_val = self.target_critic(next_obs)  # (*,)
        target = torch.where(done, reward, reward + self.gamma * next_val)  # (*,)

        values = self.critics(obs, action)  # [(*,)] * N
        return torch.stack([target - v for v in values], dim=-1)  # (*, N)

    @staticmethod
    def gradient_loss(delta: Tensor, action: Tensor) -> Tensor:
        """Returns the action gradient loss for the Q-value function."""
        (action_gradient,) = torch.autograd.grad(delta.sum(), action, create_graph=True)
        return torch.sum(action_gradient ** 2, dim=-1).mean()

    @staticmethod
    def temporal_diff_loss(delta: Tensor) -> Tensor:
        """Returns the temporal difference loss for the Q-value function."""
        return torch.sum(delta ** 2, dim=-1).mean()
