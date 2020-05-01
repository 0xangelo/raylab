# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import RolloutWorker


@pytest.fixture
def worker_kwargs():
    return {"rollout_fragment_length": 1, "batch_mode": "complete_episodes"}


def test_compute_single_action(envs, env_name, policy_cls):
    env = envs[env_name]({})
    policy = policy_cls(env.observation_space, env.action_space, {"env": env_name})

    obs = env.observation_space.sample()
    action, states, info = policy.compute_single_action(obs, [])
    assert action in env.action_space
    assert isinstance(states, list)
    assert isinstance(info, dict)


def test_policy_in_rollout_worker(envs, env_name, policy_cls, worker_kwargs):
    env_creator = envs[env_name]
    policy_config = {"env": env_name}
    worker = RolloutWorker(
        env_creator=env_creator,
        policy=policy_cls,
        policy_config=policy_config,
        **worker_kwargs
    )
    worker.sample()
