"""Tune experiment configuration to test SOP in the Industrial Benchmark."""
from ray import tune
from ib_defaults import get_config as base_config


def get_config():  # pylint: disable=missing-docstring
    return {
        **base_config(),
        # === Environment ===
        "env": "IndustrialBenchmark",
        "env_config": {
            "reward_type": "classic",
            "action_type": "continuous",
            "observation": tune.grid_search(["visible", "markovian"]),
            "max_episode_steps": 500,
            "time_aware": True,
        },
        # === Trainer ===
        "train_batch_size": tune.grid_search([32, 128]),
    }
