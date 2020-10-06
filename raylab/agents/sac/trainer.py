"""Soft Actor-Critic.

Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.
"""
from raylab.agents import Trainer
from raylab.agents.off_policy import OffPolicyMixin
from raylab.options import configure

from .policy import SACTorchPolicy


@configure
@OffPolicyMixin.add_options
class SACTrainer(OffPolicyMixin, Trainer):
    """Single agent trainer for SAC."""

    _name = "SoftAC"
    _policy_class = SACTorchPolicy
