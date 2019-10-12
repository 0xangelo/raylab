"""Debugging utilities."""
from numpy.random import uniform, randn
from ray.rllib.policy.sample_batch import SampleBatch


def fake_batch(obs_space, action_space, batch_size=1):
    """Create a fake SampleBatch compatible with Policy.learn_on_batch."""
    obs_sample_shape = (batch_size,) + obs_space.shape
    act_sample_shape = (batch_size,) + action_space.shape
    samples = {
        SampleBatch.CUR_OBS: randn(*obs_sample_shape),
        SampleBatch.ACTIONS: uniform(
            action_space.low, action_space.high, size=act_sample_shape
        ),
        SampleBatch.REWARDS: randn(batch_size),
        SampleBatch.NEXT_OBS: randn(*obs_sample_shape),
        SampleBatch.DONES: randn(batch_size) > 0,
    }
    return SampleBatch(samples)
