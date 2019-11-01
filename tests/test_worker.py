# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch


@pytest.fixture(params=[True, False])
def env_creator(request, navigation_env, cartpole_swingup_env):
    return navigation_env if request.param else cartpole_swingup_env


@pytest.fixture
def worker(env_creator, policy_cls):
    return RolloutWorker(
        env_creator=env_creator,
        policy=policy_cls,
        batch_steps=1,
        batch_mode="complete_episodes",
    )


def test_collect_traj(worker):
    traj = worker.sample()
    assert isinstance(traj, SampleBatch)
