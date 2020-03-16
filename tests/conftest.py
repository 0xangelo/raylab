# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import logging
import pytest
import gym

import raylab
from raylab.algorithms.registry import ALGORITHMS

from .mock_env import MockEnv

gym.logger.set_level(logging.ERROR)


@pytest.fixture
def envs():
    from raylab.envs.registry import ENVS  # pylint:disable=import-outside-toplevel

    ENVS["MockEnv"] = lambda config: MockEnv(config)

    raylab.register_all_environments()
    return ENVS.copy()


@pytest.fixture(params=list(ALGORITHMS.values()))
def trainer_cls(request):
    return request.param()


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy


@pytest.fixture(params={"MockEnv", "Navigation", "Reservoir", "HVAC"})
def env_name(request):
    return request.param


@pytest.fixture
def env_creator(envs, env_name):
    return envs[env_name]
