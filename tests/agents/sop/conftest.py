# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.agents.registry import AGENTS


@pytest.fixture
def sop_trainer():
    return AGENTS["SOP"]()


@pytest.fixture
def sop_policy(sop_trainer):
    return sop_trainer._policy


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_, sop_policy):
    def make_policy_and_batch(config):
        return policy_and_batch_(sop_policy, config)

    return make_policy_and_batch
