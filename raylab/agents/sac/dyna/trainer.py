"""Soft Actor-Critic with Dyna-like data augmentation for critic learning."""
from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.sac.trainer import sac_config
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import DynaSACTorchPolicy


@trainer.configure
@trainer.option("module/type", "ModelBasedSAC")
@trainer.option("torch_optimizer/models", {"type": "Adam", "lr": 1e-3})
@trainer.option("model_training", TrainingSpec().to_dict(), help=TrainingSpec.__doc__)
@trainer.option("exploration_config/pure_exploration_steps", 1000)
@trainer.option("evaluation_config/explore", False, override=True)
@sac_config
class DynaSACTrainer(ModelBasedTrainer):
    """Single agent trainer for Dyna-SAC."""

    _name = "Dyna-SAC"
    _policy = DynaSACTorchPolicy
