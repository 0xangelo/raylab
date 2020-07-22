"""Losses for computing policy gradients."""
from typing import Optional
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from torch import Tensor
from torch.autograd import grad

from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy
from raylab.policy.modules.actor.policy.stochastic import StochasticPolicy
from raylab.policy.modules.critic.q_value import QValueEnsemble
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict

from .abstract import Loss


class DeterministicPolicyGradient(Loss):
    """Loss function for Deterministic Policy Gradient.

    Args:
        actor: deterministic policy
        critics: callables for action-values
    """

    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)

    def __init__(self, actor: DeterministicPolicy, critics: QValueEnsemble):
        self.actor = actor
        self.critics = critics

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        obs = batch[SampleBatch.CUR_OBS]

        values = self.state_value(obs)
        loss = -torch.mean(values)

        stats = {"loss(actor)": loss.item()}
        return loss, stats

    def state_value(self, obs: Tensor) -> Tensor:
        """Compute the state value by combining policy and action-value function."""
        actions = self.actor(obs)
        return self.critics(obs, actions).min(dim=-1)[0]


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
        self, actor: StochasticPolicy, critics: QValueEnsemble,
    ):
        self.actor = actor
        self.critics = critics

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        obs = batch[SampleBatch.CUR_OBS]

        action_values, entropy = self.action_value_plus_entropy(obs)
        loss = -torch.mean(action_values + self.alpha * entropy)

        stats = {"loss(actor)": loss.item(), "entropy": entropy.mean().item()}
        return loss, stats

    def action_value_plus_entropy(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute the action-value and a single sample estimate of the policy's entropy.
        """
        actions, logp = self.actor.rsample(obs)
        action_values = self.critics(obs, actions).min(dim=-1)[0]
        return action_values, -logp


class ActionDPG(Loss):
    # pylint:disable=line-too-long
    """Deterministic Policy Gradient via an MSE action loss.

    Implementation based on `Acmes's DPG`_.

    .. _`Acme's DPG`: https://github.com/deepmind/acme/blob/51c4db7c8ec27e040ac52d65347f6f4ecfe04f81/acme/tf/losses/dpg.py#L21

    Args:
        actor: deterministic policy
        critics: Q-value functions

    Attributes:
        dqda_clipping: Optional value by which to clip the action gradients
        clip_norm: Whether to clip action grads by norm or value
    """
    # pylint:enable=line-too-long

    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)
    dqda_clipping: Optional[float] = None
    clip_norm: bool = True

    def __init__(self, actor: DeterministicPolicy, critics: QValueEnsemble):
        self.actor = actor
        self.critics = critics

    def compile(self):
        self.actor = torch.jit.script(self.actor)
        self.critics = torch.jit.script(self.critics)

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        obs = batch[SampleBatch.CUR_OBS]
        a_max = self.actor(obs)
        q_max = self.critics(obs, a_max).min(dim=-1)[0]

        loss, dqda_norm = self.action_dpg(
            q_max, a_max, self.dqda_clipping, self.clip_norm
        )
        loss = loss.mean()
        return loss, {"loss(actor)": loss.item(), "dqda_norm": dqda_norm.mean().item()}

    @staticmethod
    def action_dpg(
        q_max: Tensor, a_max: Tensor, dqda_clipping: Optional[float], clip_norm: bool
    ) -> Tuple[Tensor, Tensor]:
        """Deterministic policy gradient loss, similar to trfl.dpg.

        Args:
            q_max: Q-value of the approximate greedy action
            a_max: Action from the policy's output
            dqda_clipping: Optional value by which to clip the action gradients
            clip_norm: Whether to clip action grads by norm or value

        Returns:
            The DPG loss and the norm of the action-value gradient, both for
            each batch dimension
        """
        # Fake a Jacobian-vector product to calculate grads w.r.t. to batch of actions
        dqda = grad(q_max, [a_max], grad_outputs=torch.ones_like(q_max))[0]
        dqda_norm = torch.norm(dqda, dim=-1, keepdim=True)

        if dqda_clipping:
            if clip_norm:
                clip_coef = dqda_clipping / dqda_norm
                dqda = torch.where(clip_coef < 1, dqda * clip_coef, dqda)
            else:
                dqda = torch.clamp(dqda, min=-dqda_clipping, max=dqda_clipping)

        # Target_a ensures correct gradient calculated during backprop.
        target_a = dqda + a_max
        # Stop the gradient going through Q network when backprop.
        target_a = target_a.detach()
        # Gradient only go through actor network.
        loss = 0.5 * torch.sum(torch.square(target_a - a_max), dim=-1)
        # This recovers the DPG because (letting w be the actor network weights):
        # d(loss)/dw = 0.5 * (2 * (target_a - a_max) * d(target_a - a_max)/dw)
        #            = (target_a - a_max) * [d(target_a)/dw  - d(a_max)/dw]
        #            = dq/da * [d(target_a)/dw  - d(a_max)/dw]  # by defn of target_a
        #            = dq/da * [0 - d(a_max)/dw]                # by stop_gradient
        #            = - dq/da * da/dw
        return loss, dqda_norm
