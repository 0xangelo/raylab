# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.algorithms.registry import ALGORITHMS as ALGS


@pytest.fixture
def mapo_trainer():
    return ALGS["MAPO"]()


@pytest.fixture
def mapo_policy(mapo_trainer):
    return mapo_trainer._policy


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_fn, mapo_policy):
    def make_policy_and_batch(config):
        return policy_and_batch_fn(mapo_policy, config)

    return make_policy_and_batch
