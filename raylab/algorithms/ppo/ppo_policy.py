from ray.rllib.agents.ppo import ppo_policy
from ray.rllib.policy.sample_batch import SampleBatch


def should_bootstrap(policy, sample_batch, episode):
    completed = sample_batch[SampleBatch.DONES][-1]
    last_info = sample_batch.data.get(SampleBatch.INFOS, [{}])[-1]
    gym_horizon_hit = last_info.get("TimeLimit.truncated", False)
    config_horizon = policy.config["horizon"] or float("inf")
    config_horizon_hit = False if episode is None else episode.length >= config_horizon
    peb = policy.config["timeout_bootstrap"] and (gym_horizon_hit or config_horizon_hit)
    return not completed or peb


def postprocess_fn(policy, sample_batch, other_agent_batches=None, episode=None):
    """Modify the last `done` flag if Partial Episode Bootstrapping is enabled."""

    done = not should_bootstrap(policy, sample_batch, episode)
    if SampleBatch.DONES in sample_batch:
        sample_batch[SampleBatch.DONES][-1] = done

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
