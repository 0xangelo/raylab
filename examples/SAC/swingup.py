from raylab.cli.utils import tune_experiment


def policy_config():
    return {
        # === Off Policy ===
        "buffer_size": int(1e5),
        "batch_size": 128,
        "target_entropy": "auto",
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        "module": {
            "actor": {"encoder": {"units": (128, 128), "activation": "Swish"}},
            "critic": {"encoder": {"units": (128, 128), "activation": "Swish"}},
            "entropy": {"initial_alpha": 0.05},
        },
        "exploration_config": {"pure_exploration_steps": 5000},
    }


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        # === Policy ===
        "policy": policy_config(),
        # === Trainer ===
        "rollout_fragment_length": 200,
        "batch_mode": "truncate_episodes",
        "timesteps_per_iteration": 1000,
        "learning_starts": 5000,
        "evaluation_interval": 5,
        "evaluation_config": {
            "env_config": {"max_episode_steps": 1000, "time_aware": False},
        },
    }


@tune_experiment
def main():
    config = get_config()
    return (
        "SoftAC",
        config,
        {"stop": {"timesteps_total": config["policy"]["buffer_size"]}},
    )


if __name__ == "__main__":
    main()
