# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest


@pytest.fixture
def navigation_env(envs):
    return envs["Navigation"]


@pytest.fixture
def reservoir_env(envs):
    return envs["Reservoir"]


@pytest.fixture
def hvac_env(envs):
    return envs["HVAC"]
