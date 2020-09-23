import numpy as np
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "Walker2DBulletEnv-v0",
        "env_config": {"max_episode_steps": 500, "time_aware": False},
        # === Entropy ===
        # Target entropy to optimize the temperature parameter towards
        # If "auto", will use the heuristic provided in the SAC paper:
        # H = -dim(A), where A is the action space
        "target_entropy": None,
        # === Replay Buffer ===
        "buffer_size": int(4e5),
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
            "entropy": {
                "initial_alpha": tune.sample_from(
                    lambda _: np.random.uniform(0.01, 0.5)
                )
            },
        },
        # === Trainer ===
        "train_batch_size": 256,
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
            "env_config": {"max_episode_steps": 1000, "time_aware": False},
        },
    }
