from raylab.cli.utils import tune_experiment


def get_config():
    return {
        # === Environment ===
        "env": "CartPoleSwingUp-v1",
        "env_config": {"max_episode_steps": 200, "time_aware": False},
        "target_entropy": "auto",
        # === Replay Buffer ===
        "buffer_size": int(1e5),
        # === Optimization ===
        # PyTorch optimizers to use
        "optimizer": {
            "model": {"type": "Adam", "lr": 3e-4},
            "actor": {"type": "Adam", "lr": 3e-4},
            "critic": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        # === Network ===
        "module": {
            "model": {"encoder": {"units": (64, 64)}},
            "actor": {"encoder": {"units": (64, 64)}},
            "critic": {"encoder": {"units": (64, 64)}, "target_vf": True},
            "entropy": {"initial_alpha": 0.05},
        },
        "rollout_fragment_length": 200,
        "batch_mode": "truncate_episodes",
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Exploration Settings ===
        "exploration_config": {"pure_exploration_steps": 5000},
        "learning_starts": 5000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
        "evaluation_config": {
            "env_config": {"max_episode_steps": 500, "time_aware": False},
        },
    }


@tune_experiment
def main():
    config = get_config()

    return "SoftSVG", config, {"stop": {"timesteps_total": config["buffer_size"]}}


if __name__ == "__main__":
    main()
