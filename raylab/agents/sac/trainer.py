"""
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor.
"""
from raylab.agents.off_policy import SimpleOffPolicy

from .policy import SACTorchPolicy


class SACTrainer(SimpleOffPolicy):
    """Single agent trainer for SAC."""

    _name = "SoftAC"

    def get_policy_class(self, _):
        return SACTorchPolicy
