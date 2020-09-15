import pytest

from raylab.agents.registry import get_agent_cls


@pytest.fixture(scope="module")
def trainer_cls():
    return get_agent_cls("MBPO")


@pytest.fixture(scope="module")
def policy_cls(policy_fn):
    from raylab.agents.mbpo import MBPOTorchPolicy

    def make_policy(config):
        return policy_fn(MBPOTorchPolicy, config)

    return make_policy
