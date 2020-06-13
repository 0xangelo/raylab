# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest


@pytest.fixture(scope="module")
def mapo_trainer():
    from raylab.agents.registry import AGENTS

    return AGENTS["MAPO"]()


@pytest.fixture(scope="module")
def mapo_policy(mapo_trainer):
    return mapo_trainer._policy


@pytest.fixture(scope="module")
def policy_and_batch_fn(policy_and_batch_, mapo_policy):
    from ray.rllib import SampleBatch
    from raylab.envs import get_reward_fn

    def make_policy_and_batch(config):
        config["env"] = "MockEnv"
        policy, batch = policy_and_batch_(mapo_policy, config)
        reward_fn = get_reward_fn("MockEnv")
        batch[SampleBatch.REWARDS] = reward_fn(
            batch[SampleBatch.CUR_OBS],
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.NEXT_OBS],
        )
        return policy, batch

    return make_policy_and_batch
