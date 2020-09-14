"""Continuous Q-Learning with Normalized Advantage Functions."""
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import NAFTorchPolicy


class NAFTrainer(OffPolicyTrainer):
    """Single agent trainer for NAF."""

    _name = "NAF"

    def get_policy_class(self, _):
        return NAFTorchPolicy
