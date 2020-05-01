from ray import tune
from ray.rllib.utils import merge_dicts

from ib_defaults import get_config as base_config


def get_config():
    return merge_dicts(
        base_config(),
        {
            # === Environment ===
            "env": "IndustrialBenchmark",
            "env_config": {
                "reward_type": "classic",
                "action_type": "continuous",
                "observation": tune.grid_search(["visible", "markovian"]),
                "max_episode_steps": 1000,
                "time_aware": True,
            },
        },
    )
