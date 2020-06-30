# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest


@pytest.fixture(scope="module")
def trainer_cls():
    from raylab.agents.registry import AGENTS

    return AGENTS["MBPO"]()


@pytest.fixture(scope="module")
def policy_cls(policy_fn, trainer_cls):
    def make_policy(config):
        return policy_fn(trainer_cls._policy, config)

    return make_policy
