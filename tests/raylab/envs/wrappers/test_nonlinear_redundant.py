import numpy as np
import pytest

from raylab.envs.wrappers import NonlinearRedundant


@pytest.fixture
def wrapped(env):
    return NonlinearRedundant(env)


def test_observation_space(wrapped):
    base = wrapped.env.observation_space
    wrap = wrapped.observation_space

    assert wrap.dtype == base.dtype
    assert wrap.shape == (base.shape[0] * 3,)


def test_reset(wrapped):
    base = wrapped.env.observation_space

    obs = wrapped.reset()
    assert obs in wrapped.observation_space
    assert obs.shape == (base.shape[0] * 3,)


def test_step(wrapped):
    wrapped.reset()
    base = wrapped.env.observation_space

    action = wrapped.action_space.sample()
    obs, rew, done, info = wrapped.step(action)
    assert obs in wrapped.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert obs.shape == (base.shape[0] * 3,)
