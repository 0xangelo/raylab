"""Trainer and configuration for MAPO."""
from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.sac.trainer import sac_config
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MAPOTorchPolicy


@trainer.configure
@trainer.option("losses/", help="Configurations for model and actor loss functions")
@trainer.option(
    "losses/grad_estimator",
    default="PD",
    help="""Gradient estimator for optimizing expectations.

    Possible types include
    SF: score function
    PD: pathwise derivative
    """,
)
@trainer.option(
    "losses/lambda",
    default=0.0,
    help="Model KL regularization to avoid degenerate solutions (needs tuning)",
)
@trainer.option(
    "losses/manhattan",
    default=False,
    help="Whether to compute the action gradient's 1-norm or squared error",
)
@trainer.option(
    "losses/model_samples",
    default=1,
    help="Number of next states to sample from the model when calculating the"
    " model-aware deterministic policy gradient",
)
@trainer.option("module/type", default="ModelBasedSAC")
@trainer.option("torch_optimizer/models/type", default="Adam")
@trainer.option(
    "model_training", default=TrainingSpec().to_dict(), help=TrainingSpec.__doc__
)
@trainer.option(
    "model_warmup",
    default=TrainingSpec().to_dict(),
    help="""Specifications for model warm-up.

    Same configurations as 'model_training'.
    """,
)
@trainer.option("holdout_ratio", 0, override=True)
@trainer.option("max_holdout", 0, override=True)
@trainer.option("evaluation_config/explore", False, override=True)
@trainer.option("rollout_fragment_length", 25, override=True)
@sac_config
class MAPOTrainer(ModelBasedTrainer):
    """Single agent trainer for Model-Aware Policy Optimization."""

    _name = "MAPO"
    _policy = MAPOTorchPolicy
