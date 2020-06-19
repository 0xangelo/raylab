"""Trainer and configuration for MBPO."""
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.model_based import with_base_config
from raylab.policy.model_based.sampling_mixin import SamplingSpec
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MBPOTorchPolicy

DEFAULT_CONFIG = with_base_config(
    {
        # === MBPOTorchPolicy ===
        "module": {
            "type": "ModelBasedSAC",
            "model": {
                "encoder": {"units": (128, 128), "activation": "Swish"},
                "ensemble_size": 7,
                "input_dependent_scale": True,
            },
            "actor": {
                "encoder": {"units": (128, 128), "activation": "Swish"},
                "input_dependent_scale": True,
            },
            "critic": {"encoder": {"units": (128, 128), "activation": "Swish"}},
            "entropy": {"initial_alpha": 0.05},
        },
        "torch_optimizer": {
            "models": {"type": "Adam", "lr": 3e-4, "weight_decay": 0.0001},
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        # === SACTorchPolicy ===
        "target_entropy": "auto",
        "clipped_double_q": True,
        "polyak": 0.995,
        # === ModelTrainingMixin ===
        "model_training": TrainingSpec().to_dict(),
        # === ModelSamplingMixin ===
        "model_sampling": SamplingSpec().to_dict(),
        # === Policy ===
        "exploration_config": {"type": "raylab.utils.exploration.StochasticActor"},
        # === ModelBasedTrainer ===
        "holdout_ratio": 0.2,
        "model_rollouts": 20,
        "max_holdout": 5000,
        "policy_improvements": 10,
        "real_data_ratio": 0.1,
        "learning_starts": 5000,
        # === OffPolicyTrainer ===
        "train_batch_size": 512,
    }
)


class MBPOTrainer(ModelBasedTrainer):
    """Model-based trainer using SAC for policy improvement."""

    _name = "MBPO"
    _default_config = DEFAULT_CONFIG
    _policy = MBPOTorchPolicy
