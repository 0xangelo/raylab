# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest


EXPLORATION_TYPES = "ParameterNoise GaussianNoise".split(" ")


@pytest.fixture
def policy_cls():
    from raylab.agents.naf import NAFTorchPolicy

    return NAFTorchPolicy


@pytest.fixture(
    params=EXPLORATION_TYPES, ids=tuple(s.split(".")[-1] for s in EXPLORATION_TYPES),
)
def exploration(request):
    return "raylab.utils.exploration." + request.param


def test_policy_creation(policy_cls, obs_space, action_space, exploration):
    policy_cls(obs_space, action_space, {"exploration_config": {"type": exploration}})
