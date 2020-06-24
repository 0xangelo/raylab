# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import logging

import gym
import gym.spaces as spaces
import pytest

from .mock_env import MockEnv

gym.logger.set_level(logging.ERROR)


# Test setup from:
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
def pytest_addoption(parser):
    parser.addoption(
        "--skipslow", action="store_true", default=False, help="skip slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skipslow"):
        # --skipslow given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="--skipslow option passed")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture(autouse=True, scope="session")
def init_ray():
    import ray

    ray.init()
    yield
    ray.shutdown()


@pytest.fixture(autouse=True, scope="session")
def register_envs():
    # pylint:disable=import-outside-toplevel
    import raylab
    from raylab.envs.registry import ENVS

    def _mock_env_maker(config):
        return MockEnv(config)

    ENVS["MockEnv"] = _mock_env_maker
    raylab.register_all_environments()


@pytest.fixture(scope="module", params=((1,), (4,)), ids=("Obs1Dim", "Obs4Dim"))
def obs_space(request):
    return spaces.Box(-10, 10, shape=request.param)


@pytest.fixture(scope="module", params=((1,), (4,)), ids=("Act1Dim", "Act4Dim"))
def action_space(request):
    return spaces.Box(-1, 1, shape=request.param)


@pytest.fixture(scope="module")
def envs():
    from raylab.envs.registry import ENVS  # pylint:disable=import-outside-toplevel

    return ENVS.copy()


ENV_IDS = ("MockEnv", "Navigation", "Reservoir", "HVAC", "MountainCarContinuous-v0")


@pytest.fixture(params=ENV_IDS)
def env_name(request):
    return request.param


@pytest.fixture
def env_creator(envs, env_name):
    return envs[env_name]
