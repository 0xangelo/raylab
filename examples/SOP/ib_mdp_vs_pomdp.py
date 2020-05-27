from ib_defaults import get_config as base_config
from ray import tune
from ray.rllib.utils import merge_dicts


def get_config():
    return merge_dicts(
        base_config(),
        {
            # === Environment ===
            "env": "IndustrialBenchmark-v0",
            "env_config": {
                "reward_type": "classic",
                "action_type": "continuous",
                "observation": tune.grid_search(["visible", "markovian"]),
                "max_episode_steps": 1000,
                "time_aware": True,
            },
        },
    )
