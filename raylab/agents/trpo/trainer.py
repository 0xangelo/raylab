"""Trainer and configuration for TRPO."""
from ray.rllib.utils import override

from raylab.agents.trainer import Trainer

from .policy import TRPOTorchPolicy


class TRPOTrainer(Trainer):
    """Single agent trainer for TRPO."""

    _name = "TRPO"
    _policy = TRPOTorchPolicy

    @override(Trainer)
    def validate_config(self, config: dict):
        super().validate_config(config)
        assert not config[
            "learning_starts"
        ], "No point in having a warmup for an on-policy algorithm."
