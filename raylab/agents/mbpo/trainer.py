"""Trainer and configuration for MBPO."""
from raylab.agents import Trainer
from raylab.agents.model_based import ModelBasedMixin
from raylab.agents.off_policy import OffPolicyMixin
from raylab.options import configure

from .policy import MBPOTorchPolicy


@configure
@OffPolicyMixin.add_options
class MBPOTrainer(ModelBasedMixin, OffPolicyMixin, Trainer):
    """Model-based trainer using SAC for policy improvement."""

    _name = "MBPO"
    _policy_class = MBPOTorchPolicy
