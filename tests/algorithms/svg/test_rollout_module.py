import pytest
import numpy as np
import torch
from ray.rllib import RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch


@pytest.fixture(params=(True, False))
def policy_config(request):
    return {"module": {"policy": {"input_dependent_scale": request.param}}}


@pytest.fixture
def setup_navigation(svg_inf_policy, navigation_env, policy_config):
    def setup():
        worker = RolloutWorker(
            env_creator=navigation_env,
            policy=svg_inf_policy,
            policy_config=policy_config,
            batch_steps=1,
            batch_mode="complete_episodes",
        )
        policy = worker.get_policy()
        policy.set_reward_fn(worker.env.reward_fn)
        return worker, policy

    return setup


def test_reproduce_rewards(setup_navigation):
    worker, policy = setup_navigation()

    traj = worker.sample()
    tensors = policy._lazy_tensor_dict(traj)
    with torch.no_grad():
        rewards, _ = policy.module.rollout(
            tensors[SampleBatch.ACTIONS],
            tensors[SampleBatch.NEXT_OBS],
            tensors[SampleBatch.CUR_OBS][0],
        )

    assert np.allclose(traj[SampleBatch.REWARDS], rewards.numpy())


def test_propagates_gradients(setup_navigation):
    worker, policy = setup_navigation()

    traj = worker.sample()
    tensors = policy._lazy_tensor_dict(traj)
    rewards, _ = policy.module.rollout(
        tensors[SampleBatch.ACTIONS],
        tensors[SampleBatch.NEXT_OBS],
        tensors[SampleBatch.CUR_OBS][0],
    )

    rewards.sum().backward()

    assert all(p.grad is not None for p in policy.module.policy.parameters())
