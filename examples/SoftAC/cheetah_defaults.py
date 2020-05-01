"""Tune experiment configuration to test SAC in HalfCheetah-v3."""
import numpy as np
from ray import tune


def get_config():
    return {
        # === Environment ===
        "env": "HalfCheetah-v3",
        "env_config": {"max_episode_steps": 250, "time_aware": False},
        # === Replay Buffer ===
        "buffer_size": int(1e5),
        # === Network ===
        "module": {
            "actor": {
                "input_dependent_scale": True,
                "encoder": {
                    "units": (400, 300),
                    "activation": "ELU",
                    "initializer_options": {"name": "orthogonal"},
                },
            },
            "critic": {
                "encoder": {
                    "units": (400, 300),
                    "activation": "ELU",
                    "initializer_options": {"name": "orthogonal"},
                },
            },
        },
        # === Exploration Settings ===
        "exploration_config": {"pure_exploration_steps": 5000},
        # === Trainer ===
        "train_batch_size": 100,
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        "evaluation_interval": 20,
    }
