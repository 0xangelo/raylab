"""Trainer and default config for MAGE."""
from raylab.agents.model_based import SimpleModelBased

from .policy import MAGETorchPolicy


class MAGETrainer(SimpleModelBased):
    """Single agent trainer for MAGE."""

    _name = "MAGE"

    def get_policy_class(self, _):
        return MAGETorchPolicy
