import pytest


@pytest.fixture
def trainer_cls():
    from raylab.agents.registry import get_agent_cls

    return get_agent_cls("SoftAC")


@pytest.fixture
def policy_cls(trainer):
    return trainer._policy
