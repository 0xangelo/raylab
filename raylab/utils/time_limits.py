"""Utilities for handling time limits in reinforcement learning."""
from ray.rllib.policy.sample_batch import SampleBatch


def postprocess_time_limits(sample_batch, episode, horizon, time_limits):
    """Postprocess a trajectory based on the time limit type (PEB or TA)."""
    if episode and horizon and time_limits == "PEB" and episode.length >= horizon:
        sample_batch[SampleBatch.DONES][-1] = False


def should_bootstrap(sample_batch, episode, horizon, time_limits):
    completed = sample_batch[SampleBatch.DONES][-1]
    last_info = sample_batch.data.get(SampleBatch.INFOS, [{}])[-1]
    gym_horizon_hit = last_info.get("TimeLimit.truncated", False)
    config_horizon = horizon or float("inf")
    config_horizon_hit = False if episode is None else episode.length >= config_horizon
    peb = time_limits == "PEB" and (gym_horizon_hit or config_horizon_hit)
    return not completed or peb
