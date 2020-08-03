"""Losses for Model-Aware Policy Optimization."""
from typing import Tuple

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.policy.modules.actor.policy.stochastic import StochasticPolicy
from raylab.policy.modules.critic.q_value import QValueEnsemble
from raylab.policy.modules.model.stochastic.ensemble import SME
from raylab.utils.annotations import DynamicsFn
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict

from .abstract import Loss
from .mixins import EnvFunctionsMixin
from .mixins import UniformModelPriorMixin
from .utils import dist_params_stats


class MAPO(EnvFunctionsMixin, UniformModelPriorMixin, Loss):
    """Model-Aware Policy Optimization.

    Args:
        models: ensemble of stochastic models
        actor: stochastic policy
        critics: Q-value estimators

    Attributes:
        gamma: Discount factor
        alpha: Entropy regularization coefficient
    """

    batch_keys = (SampleBatch.CUR_OBS,)
    gamma: float = 0.99
    alpha: float = 0.05

    def __init__(
        self, models: SME, actor: StochasticPolicy, critics: QValueEnsemble,
    ):
        super().__init__()
        modules = nn.ModuleDict()
        modules["models"] = models
        modules["policy"] = actor
        modules["critics"] = critics
        self._modules = modules

    @property
    def initialized(self) -> bool:
        """Whether or not the loss function has all the necessary components."""
        return self._env.initialized

    @property
    def _models(self):
        return self._modules["models"]

    def compile(self):
        self._modules.update({k: torch.jit.script(m) for k, m in self._modules.items()})

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        assert self.initialized, (
            "Environment functions missing. "
            "Did you set reward and termination functions?"
        )
        obs = batch[SampleBatch.CUR_OBS]
        action, action_logp, policy_info = self._generate_action(obs)
        next_obs, obs_logp, dist_params = self.transition(obs, action)
        action_value = self.one_step_action_value_surrogate(
            obs, action, next_obs, obs_logp
        )

        entropy = -action_logp.mean()

        # Important to minimize the negative entropy
        loss = -action_value.mean() - self.alpha * entropy

        stats = {
            "loss(actor)": loss.item(),
            "entropy": entropy.item(),
        }
        stats.update(dist_params_stats(dist_params, name="model"))
        stats.update(policy_info)
        return loss, stats

    def _generate_action(self, obs: Tensor) -> Tuple[Tensor, Tensor, StatDict]:
        policy = self._modules["policy"]
        dist_params = policy(obs)
        sample, logp = policy.dist.rsample(dist_params)
        info = dist_params_stats(dist_params, name="policy")
        return sample, logp, info

    def one_step_action_value_surrogate(
        self, obs: Tensor, action: Tensor, next_obs: Tensor, log_prob: Tensor
    ) -> Tensor:
        """Surrogate loss for gradient estimation of action values.

        Computes 1-step approximation of Q^{\\pi}(s, a) for maximum entropy
        framework.

        Args:
            obs: The current observation
            action: Action taken by the agent
            next_obs: The observation resulting from the applied action
            log_prob: The log-probability of the next observation

        Returns:
            A tensor for estimating the gradient of the 1-step action-value
            function.
        """
        next_act, logp = self._modules["policy"].rsample(next_obs)

        unclipped_qval = self._modules["critics"](next_obs, next_act)
        next_qval, _ = unclipped_qval.min(dim=-1)

        reward = self._env.reward(obs, action, next_obs)
        done = self._env.termination(obs, action, next_obs)

        next_vval = (
            torch.where(done, reward, reward + self.gamma * next_qval)
            - self.alpha * logp
        )

        if self.grad_estimator == "SF":
            surrogate = log_prob * next_vval.detach()
        elif self.grad_estimator == "PD":
            surrogate = next_vval
        return surrogate


class DAPO(EnvFunctionsMixin, Loss):
    """Dynamics-Aware Policy Optimization.

    Computes the 1-step maximum entropy policy loss using a given dynamics
    function.

    Args:
        dynamics_fn: Dynamics function, usually provided by the environment
        actor: Stochastic parameterized policy
        critics: Q-value estimators

    Attributes:
        gamma: Discount factor
        alpha: Entropy regularization coefficient
        grad_estimator: Gradient estimator for expecations ('PD' or 'SF')
    """

    gamma: float = 0.99
    alpha: float = 0.05
    grad_estimator: str = "SF"

    def __init__(
        self, dynamics_fn: DynamicsFn, actor: StochasticPolicy, critics: QValueEnsemble
    ):
        super().__init__()
        self.dynamics_fn = dynamics_fn
        modules = nn.ModuleDict()
        modules["policy"] = actor
        modules["critics"] = critics
        self._modules = modules

    @property
    def initialized(self) -> bool:
        """Whether or not the loss function has all the necessary components."""
        return self._env.initialized

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        assert self.initialized, (
            "Environment functions missing. "
            "Did you set reward and termination functions?"
        )
        obs = batch[SampleBatch.CUR_OBS]
        action, action_logp = self._modules["policy"].rsample(obs)
        next_obs, obs_logp = self.transition(obs, action)
        action_value = self.one_step_action_value_surrogate(
            obs, action, next_obs, obs_logp
        )

        entropy = -action_logp.mean()
        # Important to minimize the negative entropy
        loss = -action_value.mean() - self.alpha * entropy

        stats = {
            "loss(actor)": loss.item(),
            "entropy": entropy.item(),
        }
        return loss, stats

    def transition(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute virtual transition and its log density.

        Args:
            obs: The current state
            action: The action sampled from the stochastic policy

        Returns:
            A tuple with the next state and its log-likelihood generated from
            the dynamics function
        """
        next_obs, logp = self.dynamics_fn(obs, action)
        if self.grad_estimator == "SF":
            next_obs = next_obs.detach()
        return next_obs, logp

    def one_step_action_value_surrogate(
        self, obs: Tensor, action: Tensor, next_obs: Tensor, log_prob: Tensor
    ) -> Tensor:
        """Surrogate loss for gradient estimation of action values.

        Computes 1-step approximation of Q^{\\pi}(s, a) for maximum entropy
        framework.

        Args:
            obs: The current observation
            action: Action taken by the agent
            next_obs: The observation resulting from the applied action
            log_prob: The log-probability of the next observation

        Returns:
            A tensor for estimating the gradient of the 1-step action-value
            function.
        """
        next_act, logp = self._modules["policy"].rsample(next_obs)

        unclipped_qval = self._modules["critics"](next_obs, next_act)
        next_qval, _ = unclipped_qval.min(dim=-1)

        reward = self._env.reward(obs, action, next_obs)
        done = self._env.termination(obs, action, next_obs)

        next_vval = (
            torch.where(done, reward, reward + self.gamma * next_qval)
            - self.alpha * logp
        )

        if self.grad_estimator == "SF":
            surrogate = torch.mean(log_prob * next_vval.detach(), dim=0)
        elif self.grad_estimator == "PD":
            surrogate = torch.mean(next_vval, dim=0)
        return surrogate
