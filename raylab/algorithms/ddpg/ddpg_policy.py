"""Updated DDPG policy with Partial Episode Bootstrapping."""

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.ddpg import ddpg_policy


def postprocess_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    horizon = policy.config["horizon"]
    if (
        episode
        and horizon
        and policy.config["timeout_bootstrap"]
        and episode.length >= horizon
    ):
        sample_batch[SampleBatch.DONES][-1] = False

    return ddpg_policy.postprocess_trajectory(
        policy, sample_batch, other_agent_batches, episode
    )


def get_default_config():
    """Return the default config for DDPG."""
    from raylab.algorithms.ddpg.td3 import DEFAULT_CONFIG

    return DEFAULT_CONFIG


DDPGTFPolicy = ddpg_policy.DDPGTFPolicy.with_updates(  # pylint: disable=invalid-name
    name="DDPGTFPolicy",
    get_default_config=get_default_config,
    postprocess_fn=postprocess_fn,
)
