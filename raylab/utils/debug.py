"""Debugging utilities."""
import numpy as np
from ray.rllib import SampleBatch
import gym.spaces as spaces


def fake_space_samples(space, batch_size):
    """Create fake samples from a Gym space."""
    if isinstance(space, spaces.Box):
        shape = (batch_size,) + space.shape
        return np.clip(np.random.randn(*shape), space.low, space.high)
    if isinstance(space, spaces.Discrete):
        return np.random.randint(space.n, size=batch_size)
    raise ValueError(f"Unsupported space type {type(space)}")


def fake_batch(obs_space, action_space, batch_size=1):
    """Create a fake SampleBatch compatible with Policy.learn_on_batch."""
    samples = {
        SampleBatch.CUR_OBS: fake_space_samples(obs_space, batch_size),
        SampleBatch.ACTIONS: fake_space_samples(action_space, batch_size),
        SampleBatch.REWARDS: np.random.randn(batch_size),
        SampleBatch.NEXT_OBS: fake_space_samples(obs_space, batch_size),
        SampleBatch.DONES: np.random.randn(batch_size) > 0,
    }
    return SampleBatch(samples)
