# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.algorithms.registry import ALGORITHMS as ALGS

from ...mock_env import MockReward


@pytest.fixture
def mapo_trainer():
    return ALGS["MAPO"]()


@pytest.fixture
def mapo_policy(mapo_trainer):
    return mapo_trainer._policy


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_, mapo_policy):
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
