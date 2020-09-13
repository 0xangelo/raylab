"""Trainer and default config for MAGE."""
from raylab.agents.model_based import ModelBasedTrainer
from raylab.options import configure
from raylab.options import option

from .policy import MAGETorchPolicy


@configure
@option("holdout_ratio", default=0, override=True)
@option("max_holdout", default=0, override=True)
@option("evaluation_config/explore", False, override=True)
class MAGETrainer(ModelBasedTrainer):
    """Single agent trainer for MAGE."""

    _name = "MAGE"
    _policy = MAGETorchPolicy
