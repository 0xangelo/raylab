"""Trainer and configuration for MAPO."""
from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.sac.trainer import sac_config
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MAPOTorchPolicy

DEFAULT_MODULE = {
    "type": "ModelBasedSAC",
    "model": {
        "network": {"units": (128, 128), "activation": "Swish"},
        "ensemble_size": 1,
        "input_dependent_scale": True,
        "parallelize": False,
        "residual": True,
    },
    "critic": {"double_q": True},
}


@trainer.configure
@trainer.option("losses/", help="Configurations for model and actor loss functions")
@trainer.option(
    "losses/grad_estimator",
    "SF",
    help="""Gradient estimator for optimizing expectations.

    Possible types include
    SF: score function
    PD: pathwise derivative
    """,
)
@trainer.option(
    "losses/lambda",
    0.0,
    help="Model KL regularization to avoid degenerate solutions (needs tuning)",
)
@trainer.option(
    "losses/model_samples",
    4,
    help="Number of next states to sample from the model when calculating the"
    " model-aware deterministic policy gradient",
)
@trainer.option("module", DEFAULT_MODULE, override=True)
@trainer.option("torch_optimizer/models/type", "Adam")
@trainer.option("model_training", TrainingSpec().to_dict(), help=TrainingSpec.__doc__)
@trainer.option("holdout_ratio", 0, override=True)
@trainer.option("max_holdout", 0, override=True)
@trainer.option("evaluation_config/explore", False, override=True)
@trainer.option("rollout_fragment_length", 25, override=True)
@trainer.option("batch_mode", "truncate_episodes", override=True)
@sac_config
class MAPOTrainer(ModelBasedTrainer):
    """Single agent trainer for Model-Aware Policy Optimization."""

    _name = "MAPO"
    _policy = MAPOTorchPolicy
