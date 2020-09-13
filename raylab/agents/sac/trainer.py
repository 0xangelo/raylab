"""
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
with a Stochastic Actor.
"""
from raylab.agents.off_policy import OffPolicyTrainer
from raylab.options import configure
from raylab.options import option

from .policy import SACTorchPolicy


@configure
@option("evaluation_config/explore", False, override=True)
class SACTrainer(OffPolicyTrainer):
    """Single agent trainer for SAC."""

    _name = "SoftAC"
    _policy = SACTorchPolicy
