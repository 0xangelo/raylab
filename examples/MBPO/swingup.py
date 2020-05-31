from raylab.cli.utils import tune_experiment


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        # === Entropy ===
        "target_entropy": "auto",
        # === Twin Delayed DDPG (TD3) tricks ===
        "clipped_double_q": True,
        # === Model Training ===
        "holdout_ratio": 0.2,
        "max_holdout": 5000,
        "max_model_epochs": 120,
        "max_model_steps": 120,
        "improvement_threshold": 0.01,
        "patience_epochs": 5,
        "max_model_train_s": 5,
        "model_batch_size": 256,
        # === Policy Training ===
        "policy_improvements": 10,
        "real_data_ratio": 0.1,
        "train_batch_size": 512,
        # === Replay buffer ===
        "buffer_size": int(1e5),
        "virtual_buffer_size": int(4e5),
        "model_rollouts": 40,
        "model_rollout_length": 1,
        "num_elites": 5,
        # === Optimization ===
        "torch_optimizer": {
            "models": {"type": "Adam", "lr": 1e-3, "weight_decay": 0.0001},
            "actor": {"type": "Adam", "lr": 1e-3},
            "critics": {"type": "Adam", "lr": 1e-3},
            "alpha": {"type": "Adam", "lr": 1e-3},
        },
        "polyak": 0.995,
        "learning_starts": 5000,
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
        "exploration_config": {"pure_exploration_steps": 5000},
        # === Evaluation ===
        "evaluation_interval": 2,
        "evaluation_num_episodes": 10,
        # === Common config defaults ===
        "timesteps_per_iteration": 1000,
        "rollout_fragment_length": 25,
        "batch_mode": "truncate_episodes",
        "num_cpus_for_driver": 4,
    }


@tune_experiment
def main():
    config = get_config()
    return "MBPO", config, {"stop": {"timesteps_total": config["buffer_size"]}}


if __name__ == "__main__":
    main()
