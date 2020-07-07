"""Trainer and configuration for ACKTR."""
from raylab.agents import trainer
from raylab.agents.trpo import TRPOTrainer

from .policy import ACKTRTorchPolicy
from .policy import DEFAULT_OPTIM_CONFIG


@trainer.config("torch_optimizer", DEFAULT_OPTIM_CONFIG, override=True)
@TRPOTrainer.with_base_specs
class ACKTRTrainer(TRPOTrainer):
    """Single agent trainer for ACKTR."""

    _name = "ACKTR"
    _policy = ACKTRTorchPolicy
