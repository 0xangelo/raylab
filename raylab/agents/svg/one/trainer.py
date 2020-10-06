"""Trainer and configuration for SVG(1)."""
from raylab.agents import Trainer
from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyMixin
from raylab.options import configure

from .policy import SVGOneTorchPolicy


@configure
@OffPolicyMixin.add_options
class SVGOneTrainer(OffPolicyMixin, Trainer):
    """Single agent trainer for SVG(1)."""

    _name = "SVG(1)"
    _policy_class = SVGOneTorchPolicy

    def after_init(self):
        super().after_init()
        set_policy_with_env_fn(self.workers, fn_type="reward")
