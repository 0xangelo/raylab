# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import numpy as np
import pytest
import torch

from raylab.envs import get_termination_fn
from raylab.envs.registry import ENVS
from raylab.envs.termination import TERMINATIONS


VALID_ENVS = sorted(list(set(ENVS.keys()).intersection(set(TERMINATIONS.keys()))))
ENV_CONFIGS = (
    {},
    {"max_episode_steps": 200, "time_aware": True},
)


@pytest.fixture(scope="module", params=ENV_CONFIGS, ids="TimeUnAware TimeAware".split())
def env_config(request):
    return request.param.copy()


@pytest.fixture(scope="module", params=VALID_ENVS)
def env_termination(request, envs, env_config):
    env_name = request.param
    if env_name == "IndustrialBenchmark-v0":
        env_config["max_episode_steps"] = 200

    env = envs[env_name](env_config)
    termination_fn = get_termination_fn(env_name, env_config)
    return env, termination_fn, env_config.get("time_aware", False)


def test_reproduce_terminations(env_termination):
    env, termination_fn, time_aware = env_termination

    episode, obs, done = [], env.reset(), False
    while not done:
        action = env.action_space.sample()
        new_obs, _, done, info = env.step(action)
        timeout = info.get("TimeLimit.truncated")
        terminal = False if not time_aware and timeout else done
        episode += [(obs, action, new_obs, terminal)]
        obs = new_obs

    obs, action, new_obs, done = zip(*episode)
    obs, action, new_obs, done = map(np.stack, (obs, action, new_obs, done))
    obs, action, new_obs, done = map(torch.as_tensor, (obs, action, new_obs, done))

    done_ = termination_fn(obs, action, new_obs)
    assert (~(done ^ done_)).all()
