"""Loss functions for dual variables in maximum entropy RL."""
import torch
from ray.rllib import SampleBatch


class MaximumEntropyDual:
    """Loss function for the entropy coefficient in maximum entropy RL.

    Args:
        alpha (callable): entropy coefficient
        actor (callable): stochastic policy
        target_entropy (float): minimum entropy for policy
    """

    # pylint:disable=too-few-public-methods
    ENTROPY = "entropy"

    def __init__(self, alpha, actor, target_entropy):
        self.alpha = alpha
        self.actor = actor
        self.target_entropy = target_entropy

    def __call__(self, batch):
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
