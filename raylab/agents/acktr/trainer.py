"""Trainer and configuration for ACKTR."""
from raylab.agents.trpo import TRPOTrainer

from .policy import ACKTRTorchPolicy


class ACKTRTrainer(TRPOTrainer):
    """Single agent trainer for ACKTR."""

    # pylint:disable=abstract-method
    _name = "ACKTR"

    def get_policy_class(self, _):
        return ACKTRTorchPolicy
