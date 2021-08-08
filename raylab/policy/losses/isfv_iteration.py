"""Losses for Importance Sampled Fitted V Iteration."""
from typing import Tuple

import torch
from nnrl.nn.actor import Alpha, StochasticPolicy
from nnrl.nn.critic import VValue
from nnrl.types import TensorDict
from ray.rllib import SampleBatch
from ray.rllib.utils import override
from torch import Tensor, nn

from raylab.utils.types import StatDict

from .abstract import Loss


class ISFittedVIteration(Loss):
    """Loss function for Importance Sampled Fitted V Iteration.

    Args:
        critic: state-value function
        target_critic: state-value function for the next state

    Attributes:
        gamma: discount factor
    """

    IS_RATIOS = "is_ratios"
    gamma: float = 0.99

    def __init__(self, critic: VValue, target_critic: VValue):
        self.critic = critic
        self.target_critic = target_critic
        self.batch_keys: Tuple[str, str] = (
            SampleBatch.CUR_OBS,
            self.IS_RATIOS,
            SampleBatch.NEXT_OBS,
            SampleBatch.REWARDS,
            SampleBatch.DONES,
        )

        self._loss_fn = nn.MSELoss(reduction="none")

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        """Compute loss for importance sampled fitted V iteration."""
        obs, is_ratios, next_obs, reward, done = self.unpack_batch(batch)

        with torch.no_grad():
            target = self.sampled_one_step_state_values(obs, next_obs, reward, done)

        pred = self.critic(obs)
        loss = torch.mean(is_ratios * self._loss_fn(pred, target) / 2)
        return loss, {"loss(critic)": loss.item()}

    def sampled_one_step_state_values(
        self, obs: Tensor, next_obs: Tensor, reward: Tensor, done: Tensor
    ) -> Tensor:
        """Bootstrapped approximation of true state-value using sampled transition."""
        del obs
        return torch.where(
            done, reward, reward + self.gamma * self.target_critic(next_obs)
        )


class ISSoftVIteration(ISFittedVIteration):
    """Loss function for Importance Sampled Soft V Iteration.

    Args:
        critic: state-value function
        target_critic: state-value function for the next state
        actor: stochastic policy

    Attributes:
        gamma: discount factor
    """

    # pylint:disable=too-few-public-methods
    gamma: float = 0.99

    def __init__(
        self,
        critic: VValue,
        target_critic: VValue,
        actor: StochasticPolicy,
        alpha: Alpha,
    ):
        self.actor = actor
        self.alpha = alpha
        super().__init__(critic, target_critic)

    @override(ISFittedVIteration)
    def sampled_one_step_state_values(
        self, obs: Tensor, next_obs: Tensor, reward: Tensor, done: Tensor
    ) -> Tensor:
        """Bootstrapped approximation of true state-value using sampled transition."""
        _, logp = self.actor.sample(obs)
        entropy = -logp

        reward = reward + self.alpha() * entropy
        return torch.where(
            done, reward, reward + self.gamma * self.target_critic(next_obs)
        )
