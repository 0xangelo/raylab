# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest


EXPLORATION_OPTIONS = (None, "diag_gaussian", "full_gaussian", "parameter_noise")


@pytest.fixture(
    params=EXPLORATION_OPTIONS,
    ids=tuple(
        str(e).replace("_", " ").capitalize() + " Exploration"
        for e in EXPLORATION_OPTIONS
    ),
)
def exploration(request):
    return request.param


def test_policy_creation(policy_and_batch_fn, exploration):
    policy_and_batch_fn({"exploration": exploration})
