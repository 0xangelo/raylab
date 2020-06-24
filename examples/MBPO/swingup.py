from raylab.cli.utils import tune_experiment


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
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
        # === Policy ===
        "exploration_config": {"pure_exploration_steps": 5000},
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
