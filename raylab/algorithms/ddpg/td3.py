"""Updated TD3 config with Partial Episode Bootstrapping."""
from ray.rllib.agents.ddpg import td3
from ray.rllib.agents.trainer import with_base_config
from raylab.algorithms.ddpg import ddpg_policy


DEFAULT_CONFIG = with_base_config(
    td3.TD3_DEFAULT_CONFIG,
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


TD3Trainer = td3.TD3Trainer.with_updates(  # pylint: disable=invalid-name
    name="TD3", default_policy=ddpg_policy.DDPGTFPolicy, default_config=DEFAULT_CONFIG
)
