"""Trainer and configuration for MBPO."""
from raylab.agents.model_based import DynaLikeTrainer
from raylab.agents.sac.trainer import sac_config
from raylab.options import configure
from raylab.options import option
from raylab.policy.model_based.sampling import SamplingSpec
from raylab.policy.model_based.training import TrainingSpec

from .policy import MBPOTorchPolicy

DEFAULT_MODULE = {
    "type": "ModelBasedSAC",
    "model": {
        "network": {"units": (128, 128), "activation": "Swish"},
        "ensemble_size": 7,
        "input_dependent_scale": True,
        "parallelize": True,
        "residual": True,
    },
    "actor": {
        "encoder": {"units": (128, 128), "activation": "Swish"},
        "input_dependent_scale": True,
    },
    "critic": {
        "double_q": True,
        "encoder": {"units": (128, 128), "activation": "Swish"},
    },
    "entropy": {"initial_alpha": 0.05},
}


@configure
@option("module", default=DEFAULT_MODULE, override=True)
@option(
    "torch_optimizer/models",
    default={"type": "Adam", "lr": 3e-4, "weight_decay": 0.0001},
)
@option("model_training", default=TrainingSpec().to_dict(), help=TrainingSpec.__doc__)
@option(
    "model_warmup",
    default=TrainingSpec().to_dict(),
    help="""Specifications for model warm-up.

    Same configurations as 'model_training'.
    """,
)
@option("model_sampling", default=SamplingSpec().to_dict(), help=SamplingSpec.__doc__)
@option("model_rollouts", 20, override=True)
@option("learning_starts", 5000, override=True)
@option("train_batch_size", 512, override=True)
@option("compile_policy", True, override=True)
@sac_config
class MBPOTrainer(DynaLikeTrainer):
    """Model-based trainer using SAC for policy improvement."""

    _name = "MBPO"
    _policy = MBPOTorchPolicy
