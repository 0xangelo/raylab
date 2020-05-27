# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

from raylab.agents.registry import AGENTS


@pytest.fixture(scope="module")
def mbpo_trainer():
    return AGENTS["MBPO"]()


@pytest.fixture(scope="module")
def mbpo_policy(mbpo_trainer):
    return mbpo_trainer._policy


@pytest.fixture(scope="module")
def policy_and_batch_fn(policy_and_batch_, mbpo_policy, envs):
    # pylint:disable=unused-argument
    def make_policy_and_batch(config):
        config["env"] = "MockEnv"
        return policy_and_batch_(mbpo_policy, config)

    return make_policy_and_batch
