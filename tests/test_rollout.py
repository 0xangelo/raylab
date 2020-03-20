# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import RolloutWorker


@pytest.fixture
def worker_kwargs():
    return {"batch_steps": 1, "batch_mode": "complete_episodes"}


def test_compute_single_action(env_creator, policy_cls):
    env = env_creator({})
    policy = policy_cls(env.observation_space, env.action_space, {})

    obs = env.observation_space.sample()
    action, states, info = policy.compute_single_action(obs, [])
    assert action in env.action_space
    assert isinstance(states, list)
    assert isinstance(info, dict)


def test_policy_in_rollout_worker(env_creator, policy_cls, worker_kwargs):
    worker = RolloutWorker(env_creator=env_creator, policy=policy_cls, **worker_kwargs)
    worker.sample()
