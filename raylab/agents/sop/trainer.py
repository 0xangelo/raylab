"""Trainer and configuration for SOP."""
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import SOPTorchPolicy


class SOPTrainer(OffPolicyTrainer):
    """Single agent trainer for the Streamlined Off-Policy algorithm."""

    # pytlint:disable=abstract-method
    _name = "SOP"

    def get_policy_class(self, _):
        return SOPTorchPolicy
