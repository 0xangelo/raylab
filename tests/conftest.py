# pylint: disable=missing-docstring,redefined-outer-name
import pytest

from raylab.algorithms.registry import ALGORITHMS

from .mock_env import MockEnv


@pytest.fixture(params=list(ALGORITHMS.values()))
def trainer_cls(request):
    return request.param()


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy  # pylint: disable=protected-access


@pytest.fixture
def env_creator():
    return MockEnv
