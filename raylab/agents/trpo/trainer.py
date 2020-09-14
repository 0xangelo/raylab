"""Trainer and configuration for TRPO."""
from ray.rllib.utils import override

from raylab.agents.simple_trainer import SimpleTrainer

from .policy import TRPOTorchPolicy


class TRPOTrainer(SimpleTrainer):
    """Single agent trainer for TRPO."""

    # pylint:disable=abstract-method
    _name = "TRPO"

    @staticmethod
    @override(SimpleTrainer)
    def validate_config(config: dict):
        assert not config[
            "learning_starts"
        ], "No point in having a warmup for an on-policy algorithm."

    @override(SimpleTrainer)
    def get_policy_class(self, _):
        return TRPOTorchPolicy
