"""Trainer and configuration for SVG(1)."""
from ray.rllib.utils import override

from raylab.agents.model_based import set_policy_with_env_fn
from raylab.agents.off_policy import OffPolicyTrainer

from .policy import SVGOneTorchPolicy


class SVGOneTrainer(OffPolicyTrainer):
    """Single agent trainer for SVG(1)."""

    # pylint:disable=abstract-method
    _name = "SVG(1)"

    def get_policy_class(self, _):
        return SVGOneTorchPolicy

    @override(OffPolicyTrainer)
    def after_init(self):
        super().after_init()
        set_policy_with_env_fn(self.workers, fn_type="reward")
