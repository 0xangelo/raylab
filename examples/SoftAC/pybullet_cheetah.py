import numpy as np
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "HalfCheetahBulletEnv-v0",
        "env_config": {"max_episode_steps": 250, "time_aware": False},
        # === Replay Buffer ===
        "buffer_size": int(2e5),
        # === Optimization ===
        # PyTorch optimizers to use
        "torch_optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        # === Network ===
        "module": {
            "actor": {"encoder": {"units": (128, 128)}},
            "critic": {"encoder": {"units": (128, 128)}},
        },
        # === Trainer ===
        "train_batch_size": 128,
        "timesteps_per_iteration": 1000,
        # === Exploration Settings ===
        "exploration_config": {"pure_exploration_steps": 5000},
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 10,
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {
            "env_config": {"max_episode_steps": 1000, "time_aware": False}
        },
    }
