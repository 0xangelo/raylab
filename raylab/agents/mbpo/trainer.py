"""Trainer and configuration for MBPO."""
from raylab.agents import trainer
from raylab.agents.model_based import ModelBasedTrainer
from raylab.policy.model_based.sampling_mixin import SamplingSpec
from raylab.policy.model_based.training_mixin import TrainingSpec

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

TORCH_OPTIMIZERS = {
    "models": {"type": "Adam", "lr": 3e-4, "weight_decay": 0.0001},
    "actor": {"type": "Adam", "lr": 3e-4},
    "critics": {"type": "Adam", "lr": 3e-4},
    "alpha": {"type": "Adam", "lr": 3e-4},
}


@trainer.config("module", DEFAULT_MODULE, override=True)
@trainer.config("torch_optimizer", TORCH_OPTIMIZERS, override=True)
@trainer.config(
    "target_entropy",
    "auto",
    info="Target entropy to optimize the temperature parameter towards"
    " If 'auto', will use the heuristic provided in the SAC paper,"
    " H = -dim(A), where A is the action space",
)
@trainer.config(
    "polyak",
    0.995,
    info="Interpolation factor in polyak averaging for target networks.",
)
@trainer.config("model_training", TrainingSpec().to_dict(), info=TrainingSpec.__doc__)
@trainer.config("model_sampling", SamplingSpec().to_dict(), info=SamplingSpec.__doc__)
@trainer.config(
    "exploration_config/type", "raylab.utils.exploration.StochasticActor", override=True
)
@trainer.config("model_rollouts", 20, override=True)
@trainer.config("learning_starts", 5000, override=True)
@trainer.config("train_batch_size", 512, override=True)
@trainer.config("compile_policy", True, override=True)
@ModelBasedTrainer.with_base_specs
class MBPOTrainer(ModelBasedTrainer):
    """Model-based trainer using SAC for policy improvement."""

    _name = "MBPO"
    _policy = MBPOTorchPolicy
