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
            "markovian": tune.grid_search([True, False]),
            "max_episode_steps": 500,
            "time_aware": True,
        },
        # === Network ===
        # Size and activation of the fully connected networks computing the logits
        # for the policy and action-value function. No layers means the component is
        # linear in states and/or actions.
        "module": {
            "policy": {
                "units": (64,),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
            },
            "critic": {
                "units": (64,),
                "activation": "ReLU",
                "initializer_options": {"name": "xavier_uniform"},
                "delay_action": True,
            },
        },
        # === Trainer ===
        "train_batch_size": 32,
        "timesteps_per_iteration": 1000,
    }
