"""Trainer and configuration for MBPO."""
from raylab.agents.model_based import SimpleModelBased

from .policy import MBPOTorchPolicy


class MBPOTrainer(SimpleModelBased):
    """Model-based trainer using SAC for policy improvement."""

    _name = "MBPO"

    def get_policy_class(self, _):
        return MBPOTorchPolicy
