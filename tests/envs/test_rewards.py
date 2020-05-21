# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import numpy as np
import pytest
import torch

from raylab.envs.registry import ENVS
from raylab.envs.rewards import REWARDS, get_reward_fn


VALID_ENVS = sorted(list(set(ENVS.keys()).intersection(set(REWARDS.keys()))))
ENV_CONFIGS = ({}, {"max_episode_steps": 200, "time_aware": True})


@pytest.fixture(params=ENV_CONFIGS, ids="TimeUnAware TimeAware".split())
def env_config(request):
    return request.param.copy()


@pytest.fixture(params=VALID_ENVS)
def env_reward(request, envs, env_config):
    env_name = request.param
    if "HalfCheetah" in env_name:
        env_config["exclude_current_positions_from_observation"] = False
    if "IndustrialBenchmark" in env_name:
        env_config["max_episode_steps"] = 200

    env = envs[env_name](env_config)
    reward_fn = get_reward_fn(env_name, env_config)
    return env, reward_fn


def test_reproduce_rewards(env_reward):
    env, reward_fn = env_reward

    episode, obs, done = [], env.reset(), False
    while not done:
        action = env.action_space.sample()
        new_obs, rew, done, _ = env.step(action)
        episode += [(obs, action, new_obs, rew, done)]
        obs = new_obs

    obs, action, new_obs, rew, _ = zip(*episode)
    obs, action, new_obs, rew = map(np.stack, (obs, action, new_obs, rew))
    obs, action, new_obs, rew = map(torch.Tensor, (obs, action, new_obs, rew))

    rew_ = reward_fn(obs, action, new_obs)
    assert torch.allclose(rew, rew_, atol=1e-5)
