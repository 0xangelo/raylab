"""Trainer and configuration for ACKTR."""
from raylab.agents import trainer
from raylab.agents.trpo import TRPOTrainer

from .policy import ACKTRTorchPolicy
from .policy import DEFAULT_OPTIM_CONFIG


@trainer.configure
@trainer.option("torch_optimizer", DEFAULT_OPTIM_CONFIG, override=True)
class ACKTRTrainer(TRPOTrainer):
    """Single agent trainer for ACKTR."""

    _name = "ACKTR"
    _policy = ACKTRTorchPolicy
