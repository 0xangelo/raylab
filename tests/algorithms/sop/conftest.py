import pytest

from raylab.algorithms.registry import ALGORITHMS as ALGS


@pytest.fixture
def sop_trainer():
    return ALGS["SOP"]()


@pytest.fixture
def sop_policy(sop_trainer):
    return sop_trainer._policy


@pytest.fixture
def cartpole_swingup_env(envs):
    return lambda _: envs["CartPoleSwingUp"](
        {"time_aware": True, "max_episode_steps": 200}
    )


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_fn, sop_policy):
    def make_policy_and_batch(config):
        return policy_and_batch_fn(sop_policy, config)

    return make_policy_and_batch
