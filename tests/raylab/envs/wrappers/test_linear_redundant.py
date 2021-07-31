import numpy as np
import pytest
import torch

from raylab import envs
from raylab.envs.wrappers import LinearRedundant


@pytest.fixture(params=(True, False), ids="HalfSize FullSize".split())
def size(request, env):
    original = env.observation_space.shape[0]
    return original // 2 if request.param else original


@pytest.fixture
def wrapped(env, size):
    return LinearRedundant(env, size)


@pytest.fixture
def expected_shape(env, size):
    original = env.observation_space.shape[0]
    return (original + size,)


def test_observation_space(wrapped, expected_shape):
    base = wrapped.env.observation_space
    wrap = wrapped.observation_space

    assert wrap.dtype == base.dtype
    assert wrap.shape == expected_shape


def test_seed(wrapped):
    base = wrapped.env
    seeds = base.seed() or []

    assert hasattr(wrapped, "np_random")
    if hasattr(base, "np_random"):
        assert wrapped.np_random is not base.np_random

    assert len(wrapped.seed()) == len(seeds) + 1


def test_reset(wrapped):
    obs = wrapped.reset()
    assert obs in wrapped.observation_space


def test_step(wrapped):
    wrapped.reset()
    action = wrapped.action_space.sample()
    obs, rew, done, info = wrapped.step(action)

    assert obs in wrapped.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


@pytest.fixture
def reward_fn(env_name, env_config, size):
    base = envs.get_reward_fn(env_name, env_config)
    wrapped = LinearRedundant.wrap_env_function(base, size)
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
