import numpy as np
import pytest
from ray.rllib.utils.torch_ops import convert_to_torch_tensor

from raylab.envs.wrappers import GaussianRandomWalks


@pytest.fixture(params=(1, 2, 4))
def size(request):
    return request.param


@pytest.fixture
def loc():
    return 0.0


@pytest.fixture
def scale():
    return 1.0


@pytest.fixture
def wrapped(env, size, loc, scale):
    return GaussianRandomWalks(env, size=size, loc=loc, scale=scale)


def test_spaces(wrapped, size):
    base = wrapped.env

    assert wrapped.observation_space.shape[0] == base.observation_space.shape[0] + size
    assert wrapped.action_space.shape == base.action_space.shape
    assert wrapped.observation_space.dtype == base.observation_space.dtype
    assert wrapped.action_space.dtype == base.action_space.dtype
    assert np.allclose(
        wrapped.observation_space.low[:-size], base.observation_space.low
    )
    assert np.allclose(
        wrapped.observation_space.high[:-size], base.observation_space.high
    )


def test_basic(wrapped):
    obs = wrapped.reset()
    assert obs in wrapped.observation_space

    act = wrapped.action_space.sample()
    next_obs, rew, done, info = wrapped.step(act)
    assert next_obs in wrapped.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_environment_fns(wrapped):
    obs, done = wrapped.reset(), False

    for _ in range(10):
        act = wrapped.action_space.sample()
        new_obs, rew, done, _ = wrapped.step(act)

        obs_t, act_t, new_obs_t = convert_to_torch_tensor((obs, act, new_obs))

        if hasattr(wrapped, "reward_fn"):
            rew_t = wrapped.reward_fn(obs_t, act_t, new_obs_t)
            assert np.allclose(rew, rew_t.numpy())

        if hasattr(wrapped, "termination_fn"):
            done_t = wrapped.termination_fn(obs_t, act_t, new_obs_t)
            assert done == done_t.item()

        obs = new_obs
        if done:
            obs = wrapped.reset()
            done = False
