"""Trainer and configuration for MAPO with maximum likelihood-trained model."""
from raylab.agents import trainer
from raylab.agents.mapo.trainer import DEFAULT_MODULE
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.sac.trainer import sac_config
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MlMAPOTorchPolicy


@trainer.config(
    "losses/grad_estimator",
    "SF",
    info="""\
    Gradient estimator for optimizing expectations. Possible types include
    SF: score function
    PD: pathwise derivative
    """,
)
@trainer.config(
    "losses/model_samples",
    4,
    info="""\
    Number of next states to sample from the model when calculating the
    model-aware deterministic policy gradient
    """,
)
@trainer.config(
    "losses", {}, info="Configurations for model, actor, and critic loss functions"
)
@trainer.config("module", DEFAULT_MODULE, override=True)
@trainer.config("torch_optimizer/models", {"type": "Adam", "lr": 1e-3})
@trainer.config("model_training", TrainingSpec().to_dict(), info=TrainingSpec.__doc__)
@trainer.config("evaluation_config/explore", False, override=True)
@trainer.config("rollout_fragment_length", 25, override=True)
@trainer.config("batch_mode", "truncate_episodes", override=True)
@sac_config
@ModelBasedTrainer.with_base_specs
class MlMAPOTrainer(ModelBasedTrainer):
    """Single agent trainer for MAPO-MLE."""

    _name = "MAPO-MLE"
    _policy = MlMAPOTorchPolicy
