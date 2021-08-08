"""Losses for computing policy gradients."""
from typing import Optional, Tuple, Union

import torch
from nnrl.nn.actor import Alpha, DeterministicPolicy, StochasticPolicy
from nnrl.nn.critic import ClippedQValue, QValue, QValueEnsemble
from nnrl.types import TensorDict
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.utils.types import StatDict

from .abstract import Loss
from .utils import action_dpg, dist_params_stats


def clip_if_needed(critic: Union[QValue, QValueEnsemble]) -> QValue:
    if isinstance(critic, QValueEnsemble):
        critic = ClippedQValue(critic)
    return critic


class DeterministicPolicyGradient(Loss):
    """Loss function for Deterministic Policy Gradient.

    Args:
        actor: deterministic policy
        critic: action-value function (single or ensemble)
    """

    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)

    def __init__(
        self, actor: DeterministicPolicy, critic: Union[QValue, QValueEnsemble]
    ):
        self.actor = actor
        self.critic = clip_if_needed(critic)

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        obs = batch[SampleBatch.CUR_OBS]
        act = self.actor(obs)
        val = self.critic(obs, act)
        loss = -torch.mean(val)

        stats = {"loss(actor)": loss.item()}
        return loss, stats


class ReparameterizedSoftPG(Loss):
    """Loss function for Soft Policy Iteration with reparameterized actor.

    Args:
        actor: stochastic reparameterized policy
        critic: action-value function (single or ensemble)
        alpha: entropy coefficient
    """

    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)

    def __init__(
        self,
        actor: StochasticPolicy,
        critic: Union[QValue, QValueEnsemble],
        alpha: Alpha,
    ):
        self.actor = actor
        self.critic = clip_if_needed(critic)
        self.alpha = alpha

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        obs = batch[SampleBatch.CUR_OBS]

        action_values, entropy, stats = self.action_value_plus_entropy(obs)
        loss = -torch.mean(action_values + self.alpha() * entropy)

        stats.update({"loss(actor)": loss.item(), "entropy": entropy.mean().item()})
        return loss, stats

    def action_value_plus_entropy(self, obs: Tensor) -> Tuple[Tensor, Tensor, StatDict]:
        """
        Compute the action-value and a single sample estimate of the policy's entropy.
        """
        params = self.actor(obs)
        info = dist_params_stats(params, name="policy")

        act, logp = self.actor.dist.rsample(params)
        action_values = self.critic(obs, act)
        return action_values, -logp, info


class ActionDPG(Loss):
    """Deterministic Policy Gradient by separating action and Q-value grads.

    Args:
        actor: deterministic policy
        critic: Q-value function (single or ensemble)

    Attributes:
        dqda_clipping: Optional value by which to clip the action gradients
        clip_norm: Whether to clip action grads by norm or value
    """

    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS,)
    dqda_clipping: Optional[float] = None
    clip_norm: bool = True

    def __init__(
        self, actor: DeterministicPolicy, critic: Union[QValue, QValueEnsemble]
    ):
        self.actor = actor
        self.critic = clip_if_needed(critic)

    def compile(self):
        self.actor = torch.jit.script(self.actor)
        self.critic = torch.jit.script(self.critic)

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        obs = batch[SampleBatch.CUR_OBS]
        a_max = self.actor(obs)
        q_max = self.critic(obs, a_max)

        loss, dqda_norm = action_dpg(q_max, a_max, self.dqda_clipping, self.clip_norm)
        loss = loss.mean()
        return loss, {"loss(actor)": loss.item(), "dqda_norm": dqda_norm.mean().item()}
