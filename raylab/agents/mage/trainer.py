"""Trainer and default config for MAGE."""
from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.sop.trainer import sop_config
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MAGETorchPolicy


@trainer.configure
@trainer.config("lambda", 0.05, info="TD error regularization for MAGE loss")
@trainer.config("model_training", TrainingSpec().to_dict(), info=TrainingSpec.__doc__)
@sop_config
@trainer.config("module/type", "ModelBasedDDPG")
@trainer.config("torch_optimizer/models", {"type": "Adam"})
@trainer.config(
    "exploration_config/type", "raylab.utils.exploration.GaussianNoise", override=True
)
@trainer.config("holdout_ratio", 0, override=True)
@trainer.config("max_holdout", 0, override=True)
@trainer.config("evaluation_config/explore", False, override=True)
class MAGETrainer(ModelBasedTrainer):
    """Single agent trainer for MAGE."""

    _name = "MAGE"
    _policy = MAGETorchPolicy
