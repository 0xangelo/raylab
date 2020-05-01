# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import numpy as np

from raylab.envs.wrappers import GaussianRandomWalks
from raylab.utils.pytorch import convert_to_tensor


@pytest.fixture(params=(1, 2, 4))
def env(request, env_creator):
    return GaussianRandomWalks(env_creator({}), num_walks=request.param)


def test_spaces(env):
    base_env = env.env
    walks = env._num_walks

    assert env.observation_space.shape[0] == base_env.observation_space.shape[0] + walks
    assert env.action_space.shape == base_env.action_space.shape
    assert env.observation_space.dtype == base_env.observation_space.dtype
    assert env.action_space.dtype == base_env.action_space.dtype
    assert np.allclose(
        env.observation_space.low[:-walks], base_env.observation_space.low
    )
    assert np.allclose(
        env.observation_space.high[:-walks], base_env.observation_space.high
    )


def test_basic(env):
    obs = env.reset()
    assert obs in env.observation_space

    act = env.action_space.sample()

    next_obs, rew, done, info = env.step(act)
    assert next_obs in env.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_reward_fn(env):
    if not hasattr(env, "reward_fn"):
        pytest.skip("Environment does not have a reward function. Skipping...")
    obs = env.reset()
    act = env.action_space.sample()
    _obs, rew, _, _ = env.step(act)

    obs_t, act_t, _obs_t = map(
        partial(convert_to_tensor, device="cpu"), (obs, act, _obs)
    )
    rew_t = env.reward_fn(obs_t, act_t, _obs_t)

    assert np.allclose(rew, rew_t.numpy())
