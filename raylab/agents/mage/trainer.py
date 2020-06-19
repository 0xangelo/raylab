"""Trainer and default config for MAGE."""
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.model_based import with_base_config
from raylab.policy.model_based.training_mixin import DataloaderSpec
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MAGETorchPolicy

DEFAULT_CONFIG = with_base_config(
    {
        # === ModelBasedTrainer ===
        "holdout_ratio": 0,
        "max_holdout": 0,
        "virtual_buffer_size": 0,
        "model_rollouts": 0,
        "policy_improvements": 10,
        "real_data_ratio": 1,
        # === MAGETorchPolicy ===
        # Clipped Double Q-Learning: use the minimun of two target Q functions
        # as the next action-value in the target for fitted Q iteration
        "clipped_double_q": True,
        # TD error regularization for MAGE loss
        "lambda": 0.05,
        # PyTorch optimizers to use
        "torch_optimizer": {
            "models": {"type": "Adam"},
            "actor": {"type": "Adam"},
            "critics": {"type": "Adam"},
        },
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # Update policy every this number of calls to `learn_on_batch`
        "policy_delay": 1,
        "model_training": TrainingSpec(
            dataloader=DataloaderSpec(batch_size=256, replacement=True),
            max_epochs=None,
            max_grad_steps=120,
            max_time=None,
            patience_epochs=None,
            improvement_threshold=None,
        ).to_dict(),
        "module": {"type": "ModelBasedDDPG", "model": {"ensemble_size": 1}},
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
