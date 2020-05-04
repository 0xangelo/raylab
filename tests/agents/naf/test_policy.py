# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest


EXPLORATION_TYPES = "ParameterNoise GaussianNoise".split(" ")


@pytest.fixture(
    params=EXPLORATION_TYPES, ids=tuple(s.split(".")[-1] for s in EXPLORATION_TYPES),
)
def exploration(request):
    return "raylab.utils.exploration." + request.param


def test_policy_creation(policy_and_batch_fn, exploration):
    policy_and_batch_fn({"exploration_config": {"type": exploration}})
