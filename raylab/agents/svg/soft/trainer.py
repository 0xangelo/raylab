"""Trainer and configuration for SVG(1) with maximum entropy."""
from raylab.agents import Trainer
from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyMixin
from raylab.options import configure

from .policy import SoftSVGTorchPolicy


@configure
@OffPolicyMixin.add_options
class SoftSVGTrainer(OffPolicyMixin, Trainer):
    """Single agent trainer for SoftSVG."""

    _name = "SoftSVG"
    _policy_class = SoftSVGTorchPolicy

    def after_init(self):
        super().after_init()
        set_policy_with_env_fn(self.workers, fn_type="reward")
