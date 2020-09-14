"""Trainer and configuration for SVG(1) with maximum entropy."""
from ray.rllib.utils import override

from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import SoftSVGTorchPolicy


class SoftSVGTrainer(OffPolicyTrainer):
    """Single agent trainer for SoftSVG."""

    # pylint:disable=attribute-defined-outside-init
    _name = "SoftSVG"

    def get_policy_class(self, _):
        return SoftSVGTorchPolicy

    def optimize_policy_backend(self):
        pass

    @override(OffPolicyTrainer)
    def after_init(self):
        set_policy_with_env_fn(self.workers, fn_type="reward")
        super().optimize_policy_backend()
