"""Trainer and configuration for MBPO."""
from raylab.agents.model_based import ModelBasedTrainer

from .policy import MBPOTorchPolicy


class MBPOTrainer(ModelBasedTrainer):
    """Model-based trainer using SAC for policy improvement."""

    _name = "MBPO"

    def get_policy_class(self, _):
        return MBPOTorchPolicy
