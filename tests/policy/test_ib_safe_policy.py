# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import RolloutWorker

from raylab.policy.ib_safe_policy import IBSafePolicy


@pytest.fixture(params=("visible", "markovian", "full"))
def env_config(request):
    return {"observation": request.param, "max_episode_steps": 20}


@pytest.fixture
def env_creator(envs):
    return envs["IndustrialBenchmark"]


@pytest.fixture
def policy():
    return IBSafePolicy


@pytest.fixture
def worker(env_creator, policy, env_config):
    return RolloutWorker(env_creator, policy, env_config=env_config)


def test_rollout(worker):
    worker.sample()


def test_compute_actions(worker):
    policy = worker.get_policy()
    env = worker.env
    obs_batch = [env.observation_space.sample() for _ in range(10)]
    actions, state_batches, info = policy.compute_actions(obs_batch, [])

    assert actions[0] in env.action_space
    assert isinstance(state_batches, list)
    assert isinstance(info, dict)
