"""Trainer and configuration for ACKTR."""
from raylab.agents.trpo import TRPOTrainer
from raylab.options import configure
from raylab.options import option

from .policy import ACKTRTorchPolicy
from .policy import DEFAULT_OPTIM_CONFIG


@configure
@option("torch_optimizer", DEFAULT_OPTIM_CONFIG, override=True)
class ACKTRTrainer(TRPOTrainer):
    """Single agent trainer for ACKTR."""

    # pylint:disable=abstract-method
    _name = "ACKTR"

    def get_policy_class(self, _):
        return ACKTRTorchPolicy
