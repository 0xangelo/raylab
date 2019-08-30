"""Updated DDPG policy with Partial Episode Bootstrapping."""

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.ddpg import ddpg_policy


def postprocess_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    if episode is not None and policy.config["timeout_bootstrap"]:
        done = sample_batch[SampleBatch.DONES][-1]
        sample_batch[SampleBatch.DONES][-1] = (
            False if episode.length >= policy.config["horizon"] else done
        )

    return ddpg_policy.postprocess_trajectory(
        sample_batch, other_agent_batches, episode
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
