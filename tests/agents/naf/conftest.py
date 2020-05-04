# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.agents.registry import AGENTS


@pytest.fixture
def naf_trainer():
    return AGENTS["NAF"]()


@pytest.fixture
def naf_policy(naf_trainer):
    return naf_trainer._policy


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_, naf_policy):
    def make_policy_and_batch(config):
        return policy_and_batch_(naf_policy, config)

    return make_policy_and_batch
