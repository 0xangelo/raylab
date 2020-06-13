"""Trainer and default config for MAGE."""
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.model_based import with_base_config
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MAGETorchPolicy

DEFAULT_CONFIG = with_base_config(
    {
        # === ModelBasedTrainer ===
        "virtual_buffer_size": 0,
        "model_rollouts": 0,
        # === MAGETorchPolicy ===
        # Clipped Double Q-Learning: use the minimun of two target Q functions
        # as the next action-value in the target for fitted Q iteration
        "clipped_double_q": True,
        # PyTorch optimizers to use
        "torch_optimizer": {
            "models": {"type": "Adam"},
            "actor": {"type": "Adam"},
            "critics": {"type": "Adam"},
        },
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        "model_training": TrainingSpec().to_dict(),
        "module": {"type": "MAPOModule", "model": {"ensemble_size": 1}},
        # === Exploration Settings ===
        # Default exploration behavior, iff `explore`=None is passed into
        # compute_action(s).
        # Set to False for no exploration behavior (e.g., for evaluation).
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            "type": "raylab.utils.exploration.GaussianNoise",
            "noise_stddev": 0.3,
            "pure_exploration_steps": 1000,
        },
        # === Evaluation ===
        # Extra arguments to pass to evaluation workers.
        "evaluation_config": {"explore": False},
    }
)


class MAGETrainer(ModelBasedTrainer):
    """Single agent trainer for MAGE."""

    _name = "MAGE"
    _default_config = DEFAULT_CONFIG
    _policy = MAGETorchPolicy
