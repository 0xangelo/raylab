"""Trainer and configuration for MBPO."""
from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.model_based import with_base_config
from raylab.policy.model_based.sampling_mixin import SamplingSpec
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MBPOTorchPolicy

DEFAULT_CONFIG = with_base_config(
    {
        # === Entropy ===
        # Target entropy to optimize the temperature parameter towards
        # If "auto", will use the heuristic provided in the SAC paper:
        # H = -dim(A), where A is the action space
        "target_entropy": None,
        # === Twin Delayed DDPG (TD3) tricks ===
        # Clipped Double Q-Learning
        "clipped_double_q": True,
        # === ModelBasedTrainer ===
        # Fraction of input data for validation and early stopping, may be 0.
        "holdout_ratio": 0.2,
        # Maximum number of samples for validation and early stopping
        "max_holdout": 5000,
        # Size of the buffer for virtual samples
        "virtual_buffer_size": int(1e5),
        # Number of model rollouts to add to augmented replay per real environment step
        "model_rollouts": 40,
        # Number of policy improvement steps per real environment step
        "policy_improvements": 10,
        # Fraction of each policy minibatch to sample from environment replay pool
        "real_data_ratio": 0.1,
        "train_batch_size": 512,
        # Wait until this many steps have been sampled before starting optimization.
        "learning_starts": 10000,
        # === Policy mixins ===
        # Specifications for model training
        # See `raylab.policy.model_based.training_mixin`
        "model_training": TrainingSpec().to_dict(),
        # Specifications for model sampling
        # See `raylab.policy.model_based.sampling_mixin`
        "model_sampling": SamplingSpec().to_dict(),
        # === Replay buffer ===
        # Size of the replay buffer.
        "buffer_size": int(1e5),
        # === Optimization ===
        # PyTorch optimizers to use
        "torch_optimizer": {
            "models": {"type": "Adam", "lr": 1e-3, "weight_decay": 0.0001},
            "actor": {"type": "Adam", "lr": 1e-3},
            "critics": {"type": "Adam", "lr": 1e-3},
            "alpha": {"type": "Adam", "lr": 1e-3},
        },
        # Interpolation factor in polyak averaging for target networks.
        "polyak": 0.995,
        # === Network ===
        "module": {
            "type": "ModelBasedSAC",
            "model": {
                "encoder": {"units": (128, 128), "activation": "ReLU"},
                "ensemble_size": 7,
                "input_dependent_scale": True,
            },
            "actor": {
                "encoder": {"units": (128, 128), "activation": "ReLU"},
                "input_dependent_scale": True,
            },
            "critic": {"encoder": {"units": (128, 128), "activation": "ReLU"}},
            "entropy": {"initial_alpha": 0.01},
        },
        # === Exploration Settings ===
        "explore": True,
        # Provide a dict specifying the Exploration object's config.
        "exploration_config": {
            # The Exploration class to use.
            "type": "raylab.utils.exploration.StochasticActor",
            "pure_exploration_steps": 10000,
        },
        # === Evaluation ===
        "evaluation_config": {"explore": False},
        # === Common config defaults ===
        "rollout_fragment_length": 25,
        "batch_mode": "truncate_episodes",
    }
)


class MBPOTrainer(ModelBasedTrainer):
    """Model-based trainer using SAC for policy improvement."""

    _name = "MBPO"
    _default_config = DEFAULT_CONFIG
    _policy = MBPOTorchPolicy
