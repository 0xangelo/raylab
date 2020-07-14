"""Soft Actor-Critic with Dyna-like data augmentation for critic learning."""
from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.sac.trainer import sac_config
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import DynaSACTorchPolicy


@trainer.config("module/type", "ModelBasedSAC")
@trainer.config("torch_optimizer/models", {"type": "Adam", "lr": 1e-3})
@trainer.config("model_training", TrainingSpec().to_dict(), info=TrainingSpec.__doc__)
@trainer.config("exploration_config/pure_exploration_steps", 1000)
@trainer.config("evaluation_config/explore", False, override=True)
@sac_config
@ModelBasedTrainer.with_base_specs
class DynaSACTrainer(ModelBasedTrainer):
    """Single agent trainer for Dyna-SAC."""

    _name = "Dyna-SAC"
    _policy = DynaSACTorchPolicy
