from raylab.cli.utils import tune_experiment


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        # === SACTorchPolicy ===
        "target_entropy": "auto",
        "polyak": 0.995,
        # === Policy ===
        "module": {
            "type": "ModelBasedSAC",
            "model": {
                "ensemble_size": 7,
                "parallelize": True,
                "residual": True,
                "input_dependent_scale": True,
                "network": {"units": (128, 128), "activation": "Swish"},
            },
            "actor": {
                "encoder": {"units": (128, 128), "activation": "Swish"},
                "input_dependent_scale": True,
                "initial_entropy_coeff": 0.05,
            },
            "critic": {
                "encoder": {"units": (128, 128), "activation": "Swish"},
                "double_q": True,
            },
            "initializer": {"name": "xavier_uniform"},
        },
        "optimizer": {
            "models": {"type": "Adam", "lr": 3e-4, "weight_decay": 0.0001},
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        "exploration_config": {"pure_exploration_steps": 5000},
        # === ModelTrainingMixin ===
        "model_training": {
            "dataloader": {"batch_size": 256},
            "max_epochs": 10,
            "max_grad_steps": 120,
            "max_time": 5,
            "improvement_threshold": 0.01,
            "patience_epochs": 5,
        },
        # === ModelSamplingMixin ===
        "model_sampling": {
            "num_elites": 5,
            "rollout_schedule": [(0, 1), (20000, 1), (100000, 15)],
        },
        # === ModelBasedTrainer ===
        "virtual_buffer_size": int(4e5),
        "holdout_ratio": 0.2,
        "model_rollouts": 40,
        "max_holdout": 5000,
        "policy_improvements": 20,
        "real_data_ratio": 0.1,
        # === OffPolicyTrainer ===
        "buffer_size": int(1e5),
        "train_batch_size": 512,
        "learning_starts": 5000,
        # === Trainer ===
        "evaluation_interval": 2,
        "evaluation_num_episodes": 10,
        "timesteps_per_iteration": 1000,
        "num_cpus_for_driver": 4,
        "compile_policy": True,
        # === RolloutWorker ===
        "rollout_fragment_length": 25,
        "batch_mode": "truncate_episodes",
    }


@tune_experiment
def main():
    config = get_config()
    return "MBPO", config, {"stop": {"timesteps_total": config["buffer_size"]}}


if __name__ == "__main__":
    main()
