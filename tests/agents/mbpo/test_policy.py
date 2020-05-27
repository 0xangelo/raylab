# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest


@pytest.fixture(scope="module")
def config():
    return {"module": {"ensemble_size": 1}}


@pytest.fixture(scope="module")
def policy(policy_and_batch_fn, config):
    policy, _ = policy_and_batch_fn(config)
    return policy


def test_policy_creation(policy):
    assert "models" in policy.module
    assert "actor" in policy.module
    assert "critics" in policy.module
    assert "alpha" in policy.module

    assert len(policy.optimizer) == 4
