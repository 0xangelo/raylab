from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Hopper-v3",
        "env_config": {"max_episode_steps": 1000, "time_aware": False},
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(1e5),
        # === Optimization ===
        # PyTorch optimizers to use
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        # === Network ===
        "module": {
            "actor": {"encoder": {"units": (256, 256)}},
            "critic": {"encoder": {"units": (256, 256)}},
        },
        # === Trainer ===
        "train_batch_size": 256,
        "timesteps_per_iteration": 1000,
        # === Exploration Settings ===
        "exploration_config": {"pure_exploration_steps": tune.grid_search([0, 5000])},
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 5,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 5,
    }
