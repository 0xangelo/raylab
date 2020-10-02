"""Continuous Q-Learning with Normalized Advantage Functions."""
from raylab.agents import Trainer
from raylab.agents.off_policy import OffPolicyMixin
from raylab.options import configure

from .policy import NAFTorchPolicy


@configure
@OffPolicyMixin.add_options
class NAFTrainer(OffPolicyMixin, Trainer):
    """Single agent trainer for NAF."""

    _name = "NAF"
    _policy_class = NAFTorchPolicy
