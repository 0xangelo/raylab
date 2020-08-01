"""Trainer and configuration for MAPO with maximum likelihood-trained model."""
from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.sac.trainer import sac_config
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MlMAPOTorchPolicy


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
    "losses/dyna_q",
    default=False,
    help="Whether to use the model to sample next state for CDQ-Learning.",
)
@trainer.option(
    "losses/model_samples",
    default=1,
    help="Number of next states to sample from the model when calculating the"
    " model-aware deterministic policy gradient",
)
@trainer.option("module/type", default="ModelBasedSAC")
@trainer.option("torch_optimizer/models/type", "Adam")
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
@trainer.option("evaluation_config/explore", False, override=True)
@trainer.option("rollout_fragment_length", 25, override=True)
@sac_config
class MlMAPOTrainer(ModelBasedTrainer):
    """Single agent trainer for MAPO-MLE."""

    _name = "MAPO-MLE"
    _policy = MlMAPOTorchPolicy
