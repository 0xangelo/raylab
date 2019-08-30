"""Updated TD3 config with Partial Episode Bootstrapping."""
from ray.rllib.agents.ddpg import td3
from ray.rllib.agents.trainer import with_base_config
from raylab.algorithms.ddpg import ddpg_policy


DEFAULT_CONFIG = with_base_config(
    td3.TD3_DEFAULT_CONFIG,
    {
        # Whether to ignore horizon termination and bootstrap from final observation.
        # This is used to set targets for the action value function.
        "timeout_bootstrap": True
    },
)


TD3Trainer = td3.TD3Trainer.with_updates(  # pylint: disable=invalid-name
    name="TD3", default_policy=ddpg_policy.DDPGTFPolicy, default_config=DEFAULT_CONFIG
)
