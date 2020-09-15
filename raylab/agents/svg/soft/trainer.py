"""Trainer and configuration for SVG(1) with maximum entropy."""
from ray.rllib.utils import override

from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import SoftSVGTorchPolicy


class SoftSVGTrainer(OffPolicyTrainer):
    """Single agent trainer for SoftSVG."""

    # pylint:disable=abstract-method
    _name = "SoftSVG"

    def get_policy_class(self, _):
        return SoftSVGTorchPolicy

    @override(OffPolicyTrainer)
    def after_init(self):
        super().after_init()
        set_policy_with_env_fn(self.workers, fn_type="reward")
