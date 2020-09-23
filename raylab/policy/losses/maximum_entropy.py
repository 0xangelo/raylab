"""Loss functions for dual variables in maximum entropy RL."""
from typing import Callable
from typing import Tuple

import torch
from ray.rllib import SampleBatch
from torch import Tensor

from raylab.policy.modules.actor import StochasticPolicy
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict

from .abstract import Loss


class MaximumEntropyDual(Loss):
    """Loss function for the entropy coefficient in maximum entropy RL.

    Args:
        alpha: entropy coefficient
        actor: stochastic policy
        target_entropy: minimum entropy for policy
    """

    ENTROPY = "entropy"
    batch_keys = ("entropy", SampleBatch.CUR_OBS)

    def __init__(
        self,
        alpha: Callable[[], Tensor],
        actor: StochasticPolicy,
        target_entropy: float,
    ):
        self.alpha = alpha
        self.actor = actor
        self.target_entropy = target_entropy

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        """Compute entropy coefficient loss."""

        if self.ENTROPY in batch:
            entropy = batch[self.ENTROPY]
        else:
            with torch.no_grad():
                _, logp = self.actor(batch[SampleBatch.CUR_OBS])
                entropy = -logp

        alpha = self.alpha()
        entropy_diff = torch.mean(alpha * entropy - alpha * self.target_entropy)
        info = {
            "loss(alpha)": entropy_diff.item(),
            "curr_alpha": alpha.item(),
            "entropy": entropy.mean().item(),
        }
        return entropy_diff, info
