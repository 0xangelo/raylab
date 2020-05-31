"""Trainer and configuration for MBPO."""
from ray.rllib.utils import override

from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.model_based import with_base_config

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
        # === Model Training ===
        # Fraction of input data for validation and early stopping, may be 0.
        "holdout_ratio": 0.2,
        # Maximum number of samples for validation and early stopping
        "max_holdout": 5000,
        # Maximum number of full model passes through the data, may be None.
        "max_model_epochs": None,
        # Maximum number of model gradient steps, may be None.
        "max_model_steps": 120,
        # Minimum expected relative improvement in model validation loss
        "improvement_threshold": 0.01,
        # Number of epochs to wait for any of the models to improve on the validation
        # dataset before early stopping
        "patience_epochs": 5,
        # Maximum time in seconds for training the model, may be None.
        # We check this after each epoch (not minibatch)
        "max_model_train_s": 20,
        # Size of minibatch for dynamics model training
        "model_batch_size": 256,
        # === Policy Training ===
        # Number of policy improvement steps per real environment step
        "policy_improvements": 10,
        # Fraction of each policy minibatch to sample from environment replay pool
        "real_data_ratio": 0.1,
        "train_batch_size": 512,
        # === Replay buffer ===
        # Size of the replay buffer.
        "buffer_size": int(1e5),
        # Size of the buffer for virtual samples
        "virtual_buffer_size": int(1e5),
        # Number of model rollouts to add to augmented replay per real environment step
        "model_rollouts": 40,
        # Lenght of model-based rollouts from each state sampled from replay
        "model_rollout_length": 1,
        # Use this number of best performing models on the validation dataset to sample
        # transitions
        "num_elites": 5,
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
        # Wait until this many steps have been sampled before starting optimization.
        "learning_starts": 10000,
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

    @staticmethod
    @override(ModelBasedTrainer)
    def validate_config(config):
        ModelBasedTrainer.validate_config(config)
        assert (
            config["max_model_epochs"] is None or config["max_model_epochs"] >= 0
        ), "Cannot train model for a negative number of epochs"
        assert (
            config["patience_epochs"] > 0
        ), "Must wait a positive number of epochs for any model to improve"
        assert config["model_batch_size"] > 0, "Model batch size must be positive"

        assert (
            config["model_rollout_length"] > 0
        ), "Length of model-based rollouts must be positive"

        stopping_criteria = (
            config["holdout_ratio"] > 0
            or config["max_model_epochs"]
            or config["max_model_train_s"]
        )
        assert (
            stopping_criteria
        ), "MBPO needs at least one stopping criteria for model training"
