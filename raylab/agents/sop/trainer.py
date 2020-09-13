"""Trainer and configuration for SOP."""
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.options import configure
from raylab.options import option

from .policy import SOPTorchPolicy


@configure
@option("evaluation_config/explore", False, override=True)
class SOPTrainer(OffPolicyTrainer):
    """Single agent trainer for the Streamlined Off-Policy algorithm."""

    _name = "SOP"
    _policy = SOPTorchPolicy
