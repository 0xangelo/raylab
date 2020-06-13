# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest


@pytest.fixture
def trainer():
    from raylab.agents.registry import AGENTS

    return AGENTS["SOP"]()


@pytest.fixture
def policy_cls(trainer):
    return trainer._policy


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_, policy_cls):
    def make_policy_and_batch(config):
        return policy_and_batch_(policy_cls, config)

    return make_policy_and_batch
