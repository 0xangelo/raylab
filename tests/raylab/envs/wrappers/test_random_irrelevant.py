import numpy as np
import pytest
import torch

import raylab.envs as envs
from raylab.envs.wrappers import RandomIrrelevant


@pytest.fixture(scope="module")
def size():
    return 4


@pytest.fixture(scope="module")
def loc():
    return 0.0


@pytest.fixture(scope="module")
def scale():
    return 1.0


@pytest.fixture
def wrapped(env, size, loc, scale):
    return RandomIrrelevant(env, size, loc, scale)


def test_observation_space(wrapped, size):
    base = wrapped.env.observation_space
    wrap = wrapped.observation_space

    assert wrap.dtype == base.dtype
    assert wrap.shape == (base.shape[0] + size,)


def test_seed(wrapped):
    base = wrapped.env
    seeds = base.seed() or []

    assert hasattr(wrapped, "np_random")
    if hasattr(base, "np_random"):
        assert wrapped.np_random is not base.np_random

    assert len(wrapped.seed()) == len(seeds) + 1


def test_reset(wrapped, size):
    base = wrapped.env.observation_space

    obs = wrapped.reset()
    assert obs in wrapped.observation_space
    assert obs.shape == (base.shape[0] + size,)


def test_step(wrapped, size):
    wrapped.reset()
    base = wrapped.env.observation_space

    action = wrapped.action_space.sample()
    obs, rew, done, info = wrapped.step(action)
    assert obs in wrapped.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert obs.shape == (base.shape[0] + size,)


@pytest.fixture
def reward_fn(env_name, size, env_config):
    base = envs.get_reward_fn(env_name, env_config)
    wrapped = RandomIrrelevant.wrap_env_function(base, size)
    return wrapped


@pytest.fixture
def termination_fn(env_name, size, env_config):
    base = envs.get_termination_fn(env_name, env_config)
    wrapped = RandomIrrelevant.wrap_env_function(base, size)
    return wrapped


def test_wrapped_reward_fn(wrapped, reward_fn):
    done = True
    for _ in range(10):
        if done:
            obs = wrapped.reset()
            done = False

        action = wrapped.action_space.sample()
        new_obs, rew, done, _ = wrapped.step(action)

        rew_ = reward_fn(*map(torch.from_numpy, (obs, action, new_obs))).item()
        assert np.allclose(rew, rew_, atol=1e-5)

        obs = new_obs
