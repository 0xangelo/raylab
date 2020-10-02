"""Trainer for TD3."""
from raylab.agents import Trainer
from raylab.agents.off_policy import OffPolicyMixin
from raylab.options import configure

from .policy import TD3TorchPolicy


@configure
@OffPolicyMixin.add_options
class TD3Trainer(OffPolicyMixin, Trainer):
    """Trainer for the Twin Delayed Deep Deterministic Policy Gradient algorithm."""

    _name = "TD3"
    _policy_class = TD3TorchPolicy
