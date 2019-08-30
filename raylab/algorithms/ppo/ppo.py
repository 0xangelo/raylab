"""Partial Episode Bootstrapping in PPO.
Based on:
http://proceedings.mlr.press/v80/pardo18a.html
"""
from ray.rllib.agents.trainer import with_base_config
from ray.rllib.agents.ppo import ppo

from raylab.algorithms.ppo import ppo_policy


DEFAULT_CONFIG = with_base_config(
    ppo.DEFAULT_CONFIG,
    {
        # Whether to ignore horizon termination and bootstrap from final observation.
        # This is used in GAE to set targets for the value function.
        "timeout_bootstrap": True
    },
)


PPOTrainer = ppo.PPOTrainer.with_updates(  # pylint: disable=invalid-name
    name="PPO", default_config=DEFAULT_CONFIG, default_policy=ppo_policy.PPOTFPolicy
)
