"""Trainer and default config for MAGE."""
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.sop.trainer import sop_config
from raylab.options import configure
from raylab.options import option
from raylab.policy.model_based.training import TrainingSpec

from .policy import MAGETorchPolicy


@configure
@option("lambda", default=0.05, help="TD error regularization for MAGE loss")
@option("model_training", default=TrainingSpec().to_dict(), help=TrainingSpec.__doc__)
@option(
    "model_warmup",
    default=TrainingSpec().to_dict(),
    help="""Specifications for model warm-up.

    Same configurations as 'model_training'.
    """,
)
@sop_config
@option("module/type", "ModelBasedDDPG")
@option("torch_optimizer/models", {"type": "Adam"})
@option(
    "exploration_config/type", "raylab.utils.exploration.GaussianNoise", override=True
)
@option("holdout_ratio", default=0, override=True)
@option("max_holdout", default=0, override=True)
@option("evaluation_config/explore", False, override=True)
class MAGETrainer(ModelBasedTrainer):
    """Single agent trainer for MAGE."""

    _name = "MAGE"
    _policy = MAGETorchPolicy
