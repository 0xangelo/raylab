"""Continuous Q-Learning with Normalized Advantage Functions."""
from raylab.agents.off_policy import SimpleOffPolicy

from .policy import NAFTorchPolicy


class NAFTrainer(SimpleOffPolicy):
    """Single agent trainer for NAF."""

    _name = "NAF"

    def get_policy_class(self, _):
        return NAFTorchPolicy
