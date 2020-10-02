"""Trainer and configuration for TRPO."""
from raylab.agents.trainer import Trainer

from .policy import TRPOTorchPolicy


class TRPOTrainer(Trainer):
    """Single agent trainer for TRPO."""

    _name = "TRPO"
    _policy_class = TRPOTorchPolicy
