import gym.spaces as spaces
import pytest


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


@pytest.fixture(
    params="""
    MockEnv
    Navigation
    Reservoir
    HVAC
    MountainCarContinuous-v0
    """.split(),
    scope="module",
)
def env_name(request):
    return request.param
