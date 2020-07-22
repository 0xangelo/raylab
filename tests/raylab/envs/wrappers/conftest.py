import pytest

from raylab.envs import get_env_creator


@pytest.fixture(scope="function")
def env_creator(env_name):
    return get_env_creator(env_name)


@pytest.fixture(scope="function")
def env(env_creator):
    return env_creator({})
