"""Trainer and default config for MAGE."""
from raylab.agents import Trainer
from raylab.agents.model_based import ModelBasedMixin
from raylab.agents.off_policy import OffPolicyMixin
from raylab.options import configure

from .policy import MAGETorchPolicy


@configure
@OffPolicyMixin.add_options
class MAGETrainer(ModelBasedMixin, OffPolicyMixin, Trainer):
    """Single agent trainer for MAGE."""

    _name = "MAGE"
    _policy_class = MAGETorchPolicy
