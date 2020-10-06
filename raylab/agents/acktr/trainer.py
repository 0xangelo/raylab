"""Trainer and configuration for ACKTR."""
from raylab.agents.trpo import TRPOTrainer

from .policy import ACKTRTorchPolicy


class ACKTRTrainer(TRPOTrainer):
    """Single agent trainer for ACKTR."""

    _name = "ACKTR"
    _policy_class = ACKTRTorchPolicy
