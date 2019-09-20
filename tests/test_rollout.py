# pylint: disable=missing-docstring,redefined-outer-name
import pytest
from ray.rllib import RolloutWorker

from raylab.algorithms.svg.svg_inf_policy import SVGInfTorchPolicy


@pytest.fixture
def worker_kwargs():
    return {"batch_steps": 20, "batch_mode": "complete_episodes"}


def test_output_action_in_action_space(env_creator, policy_cls):
    if policy_cls is SVGInfTorchPolicy:
        pytest.skip("SVGInfTorchPolicy doesn't squash actions to the action space.")
    env = env_creator()
    policy = policy_cls(env.observation_space, env.action_space, {})

    obs = env.observation_space.sample()
    action, _, _ = policy.compute_single_action(obs, [])
    assert action in env.observation_space


def test_policy_in_rollout_worker(env_creator, policy_cls, worker_kwargs):
    worker = RolloutWorker(
        env_creator=lambda _: env_creator(), policy=policy_cls, **worker_kwargs
    )
    worker.sample()
