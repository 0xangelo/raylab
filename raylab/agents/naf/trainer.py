"""Continuous Q-Learning with Normalized Advantage Functions."""
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.options import configure
from raylab.options import option

from .policy import NAFTorchPolicy


@configure
@option("evaluation_config/explore", False, override=True)
class NAFTrainer(OffPolicyTrainer):
    """Single agent trainer for NAF."""

    _name = "NAF"
    _policy = NAFTorchPolicy
