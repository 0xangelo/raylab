"""Trainer and configuration for SOP."""
from raylab.agents import Trainer
from raylab.agents.off_policy import OffPolicyMixin
from raylab.options import configure

from .policy import SOPTorchPolicy


@configure
@OffPolicyMixin.add_options
class SOPTrainer(OffPolicyMixin, Trainer):
    """Single agent trainer for the Streamlined Off-Policy algorithm."""

    _name = "SOP"
    _policy_class = SOPTorchPolicy
