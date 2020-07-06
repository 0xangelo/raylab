"""Trainer and default config for MAGE."""
from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.policy.model_based.training_mixin import DataloaderSpec
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MAGETorchPolicy

MODEL_TRAINING_SPEC = TrainingSpec(
    dataloader=DataloaderSpec(batch_size=256, replacement=True),
    max_epochs=None,
    max_grad_steps=120,
    max_time=None,
    patience_epochs=None,
    improvement_threshold=None,
).to_dict()
DEFAULT_MODULE = {"type": "ModelBasedDDPG", "critic": {"double_q": True}}
TORCH_OPTIMIZERS = {
    "models": {"type": "Adam"},
    "actor": {"type": "Adam"},
    "critics": {"type": "Adam"},
}
EXPLORATION_CONFIG = {
    "type": "raylab.utils.exploration.GaussianNoise",
    "noise_stddev": 0.3,
    "pure_exploration_steps": 1000,
}


@trainer.config("lambda", 0.05, info="TD error regularization for MAGE loss")
@trainer.config(
    "polyak",
    0.995,
    info="Interpolation factor in polyak averaging for target networks.",
)
@trainer.config(
    "policy_delay",
    1,
    info="Update policy every this number of calls to `learn_on_batch`",
)
@trainer.config(
    "model_training",
    MODEL_TRAINING_SPEC,
    info="See raylab.policy.model_based.ModelTrainingMixin",
)
@trainer.config("module", DEFAULT_MODULE, override=True)
@trainer.config("torch_optimizer", TORCH_OPTIMIZERS, override=True)
@trainer.config("exploration_config", EXPLORATION_CONFIG, override=True)
@trainer.config("holdout_ratio", 0, override=True)
@trainer.config("max_holdout", 0, override=True)
@trainer.config("virtual_buffer_size", 0, override=True)
@trainer.config("model_rollouts", 0, override=True)
@trainer.config("real_data_ratio", 1, override=True)
@trainer.config("evaluation_config/explore", False, override=True)
@ModelBasedTrainer.with_base_specs
class MAGETrainer(ModelBasedTrainer):
    """Single agent trainer for MAGE."""

    _name = "MAGE"
    _policy = MAGETorchPolicy
