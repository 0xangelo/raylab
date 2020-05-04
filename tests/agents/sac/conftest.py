# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.agents.registry import AGENTS


@pytest.fixture
def sac_trainer():
    return AGENTS["SoftAC"]()


@pytest.fixture
def sac_policy(sac_trainer):
    return sac_trainer._policy


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_, sac_policy):
    def make_policy_and_batch(config):
        return policy_and_batch_(sac_policy, config)

    return make_policy_and_batch
