# pylint: disable=missing-docstring,redefined-outer-name
import pytest

import raylab
from raylab.algorithms.registry import ALGORITHMS

from .mock_env import MockEnv


raylab.register_all_agents()
raylab.register_all_environments()


@pytest.fixture(params=list(ALGORITHMS.values()))
def trainer_cls(request):
    return request.param()


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy  # pylint: disable=protected-access


@pytest.fixture
def env_creator():
    return MockEnv
