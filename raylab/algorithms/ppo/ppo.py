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
        # === Time Limits ===
        # How to treat timeout terminations. Possible types include
        # None: do nothing
        # PEB: Partial Episode Bootstrapping, or bootstrap from final observation
        # TA: Time Awareness, or append relative timestep to observations
        # This is used to set targets for the action value function.
        "time_limits": "PEB"
    },
)


PPOTrainer = ppo.PPOTrainer.with_updates(  # pylint: disable=invalid-name
    name="PPO", default_config=DEFAULT_CONFIG, default_policy=ppo_policy.PPOTFPolicy
)
