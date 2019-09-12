"""Updated DDPG policy with Partial Episode Bootstrapping."""
from ray.rllib.agents.ddpg import ddpg_policy

from raylab.utils.time_limits import postprocess_time_limits


def postprocess_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    postprocess_time_limits(
        sample_batch, episode, policy.config["horizon"], policy.config["time_limits"]
    )

    return ddpg_policy.postprocess_trajectory(
        policy, sample_batch, other_agent_batches, episode
    )


def get_default_config():
    """Return the default config for DDPG."""
    # pylint: disable=cyclic-import
    from raylab.algorithms.ddpg.td3 import DEFAULT_CONFIG

    return DEFAULT_CONFIG


DDPGTFPolicy = ddpg_policy.DDPGTFPolicy.with_updates(  # pylint: disable=invalid-name
    name="DDPGTFPolicy",
    get_default_config=get_default_config,
    postprocess_fn=postprocess_fn,
)
