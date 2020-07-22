import numpy as np
import pytest

from raylab.envs import get_env_creator
from raylab.envs.wrappers import LinearRedundant


@pytest.fixture(scope="module")
def env_creator(env_name):
    return get_env_creator(env_name)


@pytest.fixture(scope="module")
def env(env_creator):
    return env_creator({})


@pytest.fixture(scope="module")
def wrapped(env):
    return LinearRedundant(env)


def test_observation_space(wrapped):
    base = wrapped.env.observation_space
    wrap = wrapped.observation_space

    assert wrap.dtype == base.dtype
    assert wrap.shape == (base.shape[0] * 2,)


def test_seed(wrapped):
    base = wrapped.env
    seeds = base.seed() or []

    assert hasattr(wrapped, "np_random")
    if hasattr(base, "np_random"):
        assert wrapped.np_random is not base.np_random

    assert len(wrapped.seed()) == len(seeds) + 1


def test_reset(wrapped):
    base = wrapped.env.observation_space

    obs = wrapped.reset()
    assert obs in wrapped.observation_space
    assert obs.shape == (base.shape[0] * 2,)


def test_step(wrapped):
    base = wrapped.env.observation_space

    action = wrapped.action_space.sample()
    obs, rew, done, info = wrapped.step(action)
    assert obs in wrapped.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert obs.shape == (base.shape[0] * 2,)
