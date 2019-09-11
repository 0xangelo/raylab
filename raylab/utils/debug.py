"""Debugging utilities."""

from numpy.random import randn
from ray.rllib.policy.sample_batch import SampleBatch


def fake_batch(obs_space, action_space, batch_size=1):
    """Create a fake SampleBatch compatible with Policy.learn_on_batch."""
    return SampleBatch(
        {
            SampleBatch.CUR_OBS: randn(batch_size, *obs_space.shape),
            SampleBatch.ACTIONS: randn(batch_size, *action_space.shape),
            SampleBatch.REWARDS: randn(batch_size),
            SampleBatch.NEXT_OBS: randn(batch_size, *obs_space.shape),
            SampleBatch.DONES: randn(batch_size) > 0,
        }
    )
