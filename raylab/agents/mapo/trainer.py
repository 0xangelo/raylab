"""Trainer and configuration for MAPO."""
from ray.rllib.utils import override

from raylab.agents.model_based import ModelBasedTrainer
from raylab.agents.model_based import with_base_config
from raylab.policy.model_based.training_mixin import DataloaderSpec
from raylab.policy.model_based.training_mixin import TrainingSpec

from .policy import MAPOTorchPolicy


DEFAULT_CONFIG = with_base_config(
    {
        # === MAPOTorchPolicy ===
        "module": {"type": "ModelBasedSAC", "model": {"ensemble_size": 1}},
        "losses": {
            # Gradient estimator for optimizing expectations. Possible types include
            # SF: score function
            # PD: pathwise derivative
            "grad_estimator": "SF",
            # KL regularization to avoid degenerate solutions (needs tuning)
            "lambda": 0.0,
            # Number of next states to sample from the model when calculating the
            # model-aware deterministic policy gradient
            "model_samples": 4,
            # Whether to use the environment's true model to sample states
            "true_model": False,
        },
        # PyTorch optimizers to use
        "torch_optimizer": {
            "models": {"type": "Adam", "lr": 1e-3},
            "actor": {"type": "Adam", "lr": 1e-3},
            "critics": {"type": "Adam", "lr": 1e-3},
            "alpha": {"type": "Adam", "lr": 1e-3},
        },
        # === SACTorchPolicy ===
        "target_entropy": "auto",
        "clipped_double_q": True,
        # === TargetNetworksMixin ===
        "polyak": 0.995,
        # === ModelTrainingMixin ===
        "model_training": TrainingSpec(
            dataloader=DataloaderSpec(batch_size=256),
            max_epochs=10,
            max_grad_steps=120,
            max_time=5,
            improvement_threshold=0.01,
            patience_epochs=5,
        ).to_dict(),
        # === Policy ===
        "explore": True,
        "exploration_config": {"type": "raylab.utils.exploration.StochasticActor"},
        # === ModelBasedTrainer ===
        "policy_improvements": 10,
        "holdout_ratio": 0,
        "max_holdout": 0,
        "virtual_buffer_size": 0,
        "model_rollouts": 0,
        "real_data_ratio": 1,
        # === OffPolicyTrainer ===
        "buffer_size": 500000,
        "learning_starts": 0,
        # === Trainer ===
        "evaluation_config": {"explore": False},
        # === Rollout Worker ===
        "rollout_fragment_length": 25,
        "batch_mode": "truncate_episodes",
    }
)


class MAPOTrainer(ModelBasedTrainer):
    """Single agent trainer for Model-Aware Policy Optimization."""

    # pylint: disable=attribute-defined-outside-init

    _name = "MAPO"
    _default_config = DEFAULT_CONFIG
    _policy = MAPOTorchPolicy

    @staticmethod
    def validate_config(config):
        constants = {
            "holdout_ratio": 0,
            "max_holdout": 0,
            "virtual_buffer_size": 0,
            "model_rollouts": 0,
            "real_data_ratio": 1,
        }
        config.update(constants)
        ModelBasedTrainer.validate_config(config)

    @override(ModelBasedTrainer)
    def _init(self, config, env_creator):
        super()._init(config, env_creator)
        policy = self.get_policy()
        if config["losses"]["true_model"]:
            worker = self.workers.local_worker()
            policy.set_dynamics_from_callable(worker.env.transition_fn)
