"""Trainer and configuration for MBPO."""
from raylab.agents.model_based import DynaLikeTrainer
from raylab.options import configure
from raylab.options import option

from .policy import MBPOTorchPolicy


@configure
@option("model_rollouts", 20, override=True)
@option("learning_starts", 5000, override=True)
@option("train_batch_size", 512, override=True)
class MBPOTrainer(DynaLikeTrainer):
    """Model-based trainer using SAC for policy improvement."""

    _name = "MBPO"
    _policy = MBPOTorchPolicy
