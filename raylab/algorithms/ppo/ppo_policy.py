"""Updated PPO policy with Partial Episode Bootstrapping."""
from ray.rllib.agents.ppo import ppo_policy

from raylab.utils.time_limits import postprocess_time_limits


def postprocess_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    """Modify the last `done` flag if Partial Episode Bootstrapping is enabled."""
    postprocess_time_limits(
        sample_batch, episode, policy.config["horizon"], policy.config["time_limits"]
    )

    return ppo_policy.postprocess_ppo_gae(
        policy, sample_batch, other_agent_batches=other_agent_batches, episode=episode
    )


def get_default_config():
    """Return the default config for PPO."""
    # pylint: disable=cyclic-import
    from raylab.algorithms.ppo.ppo import DEFAULT_CONFIG

    return DEFAULT_CONFIG


PPOTFPolicy = ppo_policy.PPOTFPolicy.with_updates(  # pylint: disable=invalid-name
    name="PPOTFPolicy",
    get_default_config=get_default_config,
    postprocess_fn=postprocess_fn,
)
