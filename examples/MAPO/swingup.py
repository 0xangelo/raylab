from raylab.cli.utils import tune_experiment


def get_config():
    return {
        # === Environment ===
        "env": "TorchCartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        # === MAPOTorchPolicy ===
        "module": {
            "type": "ModelBasedSAC",
            "model": {
                "ensemble_size": 1,
                "encoder": {"units": (128, 128), "activation": "Swish"},
            },
            "actor": {"encoder": {"units": (128, 128), "activation": "Swish"}},
            "critic": {"encoder": {"units": (128, 128), "activation": "Swish"}},
            "entropy": {"initial_alpha": 0.05},
        },
        "losses": {
            # Gradient estimator for optimizing expectations. Possible types include
            # SF: score function
            # PD: pathwise derivative
            "grad_estimator": "PD",
            # KL regularization to avoid degenerate solutions (needs tuning)
            "lambda": 0.01,
            # Number of next states to sample from the model when calculating the
            # model-aware deterministic policy gradient
            "model_samples": 1,
            # Whether to use the environment's true model to sample states
            "true_model": True,
        },
        # PyTorch optimizers to use
        "torch_optimizer": {
            "models": {"type": "Adam", "lr": 1e-4},
            "actor": {"type": "Adam", "lr": 1e-4},
            "critics": {"type": "Adam", "lr": 1e-4},
            "alpha": {"type": "Adam", "lr": 1e-4},
        },
        # === SACTorchPolicy ===
        "target_entropy": "auto",
        "clipped_double_q": True,
        # === TargetNetworksMixin ===
        "polyak": 0.995,
        # === ModelTrainingMixin ===
        "model_training": {
            "dataloader": {"batch_size": 256, "replacement": True},
            "max_epochs": 10,
            "max_grad_steps": 120,
            "max_time": 5,
            "improvement_threshold": None,
        },
        # === Policy ===
        "exploration_config": {"pure_exploration_steps": 5000},
        # === ModelBasedTrainer ===
        "policy_improvements": 10,
        "holdout_ratio": 0,
        "max_holdout": 0,
        "virtual_buffer_size": 0,
        "model_rollouts": 0,
        "real_data_ratio": 1,
        # === OffPolicyTrainer ===
        "buffer_size": int(1e5),
        "learning_starts": 5000,
        # === Trainer ===
        "timesteps_per_iteration": 500,
        "evaluation_interval": 10,
        # === Rollout Worker ===
        "rollout_fragment_length": 25,
        "batch_mode": "truncate_episodes",
    }


@tune_experiment
def main():
    config = get_config()
    return "MAPO", config, {"stop": {"timesteps_total": config["buffer_size"]}}


if __name__ == "__main__":
    main()
