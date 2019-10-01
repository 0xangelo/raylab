# pylint: disable=missing-docstring,redefined-outer-name
import pytest

import raylab
from raylab.algorithms.registry import ALGORITHMS

from .mock_env import MockEnv


@pytest.fixture
def envs():
    from raylab.envs.registry import ENVS

    raylab.register_all_environments()
    return ENVS


@pytest.fixture(params=list(ALGORITHMS.values()))
def trainer_cls(request):
    return request.param()


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy


@pytest.fixture
def env_creator():
    return MockEnv


@pytest.fixture
def navigation_env(envs):
    return envs["Navigation"]


@pytest.fixture
def time_limited_env(envs):
    return envs["TimeLimitedEnv"]


@pytest.fixture(params=(True, False))
def cartpole_swingup_env(request, time_limited_env):
    return lambda _: time_limited_env(
        {
            "env_id": "CartPoleSwingUp",
            "time_aware": request.param,
            "max_episode_steps": 200,
        }
    )
