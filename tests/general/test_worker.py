# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import RolloutWorker
from ray.rllib import SampleBatch


@pytest.fixture
def worker(envs, env_name, policy_cls):
    return RolloutWorker(
        env_creator=envs[env_name],
        policy=policy_cls,
        policy_config={"env": env_name},
        rollout_fragment_length=1,
        batch_mode="complete_episodes",
    )


def test_collect_traj(worker):
    traj = worker.sample()
    assert isinstance(traj, SampleBatch)
