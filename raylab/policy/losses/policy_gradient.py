"""Losses for computing policy gradients."""
from typing import Optional
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy
from raylab.policy.modules.actor.policy.stochastic import StochasticPolicy
from raylab.policy.modules.critic.q_value import QValueEnsemble
from raylab.utils.annotations import StatDict
from raylab.utils.annotations import TensorDict

from .abstract import Loss
from .utils import action_dpg
from .utils import dist_params_stats


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
        return QValueEnsemble.clipped(self.critics(obs, actions))


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

        action_values, entropy, stats = self.action_value_plus_entropy(obs)
        loss = -torch.mean(action_values + self.alpha * entropy)

        stats.update({"loss(actor)": loss.item(), "entropy": entropy.mean().item()})
        return loss, stats

    def action_value_plus_entropy(self, obs: Tensor) -> Tuple[Tensor, Tensor, StatDict]:
        """
        Compute the action-value and a single sample estimate of the policy's entropy.
        """
        dist_params = self.actor(obs)
        info = dist_params_stats(dist_params, name="policy")

        actions, logp = self.actor.dist.rsample(dist_params)
        action_values = QValueEnsemble.clipped(self.critics(obs, actions))
        return action_values, -logp, info


class ActionDPG(Loss):
    """Deterministic Policy Gradient by separating action and Q-value grads.

    Args:
        actor: deterministic policy
        critics: Q-value functions

    Attributes:
        dqda_clipping: Optional value by which to clip the action gradients
        clip_norm: Whether to clip action grads by norm or value
    """

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
        q_max = QValueEnsemble.clipped(self.critics(obs, a_max))

        loss, dqda_norm = action_dpg(q_max, a_max, self.dqda_clipping, self.clip_norm)
        loss = loss.mean()
        return loss, {"loss(actor)": loss.item(), "dqda_norm": dqda_norm.mean().item()}
