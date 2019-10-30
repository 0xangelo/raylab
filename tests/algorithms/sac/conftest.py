# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.algorithms.registry import ALGORITHMS as ALGS


@pytest.fixture
def sac_trainer():
    return ALGS["SAC"]()


@pytest.fixture
def sac_policy(sac_trainer):
    return sac_trainer._policy


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_fn, sac_policy):
    def make_policy_and_batch(config):
        return policy_and_batch_fn(sac_policy, config)

    return make_policy_and_batch
