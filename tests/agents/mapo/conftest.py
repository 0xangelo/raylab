# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import SampleBatch

from raylab.agents.registry import AGENTS

from ...mock_env import MockReward


@pytest.fixture(scope="module")
def mapo_trainer():
    return AGENTS["MAPO"]()


@pytest.fixture(scope="module")
def mapo_policy(mapo_trainer):
    return mapo_trainer._policy


@pytest.fixture(scope="module")
def policy_and_batch_fn(policy_and_batch_, mapo_policy, envs):
    # pylint:disable=unused-argument
    def make_policy_and_batch(config):
        config["env"] = "MockEnv"
        policy, batch = policy_and_batch_(mapo_policy, config)
        reward_fn = MockReward({})
        batch[SampleBatch.REWARDS] = reward_fn(
            batch[SampleBatch.CUR_OBS],
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.NEXT_OBS],
        )
        return policy, batch

    return make_policy_and_batch
