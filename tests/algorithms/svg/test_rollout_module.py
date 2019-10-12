import pytest
import torch
from ray.rllib import RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch


@pytest.fixture(params=(True, False))
def policy_config(request):
    return {"module": {"policy": {"input_dependent_scale": request.param}}}


@pytest.fixture
def setup_worker(svg_inf_policy, env_creator, policy_config):
    def setup():
        worker = RolloutWorker(
            env_creator=env_creator,
            policy=svg_inf_policy,
            policy_config=policy_config,
            batch_steps=1,
            batch_mode="complete_episodes",
        )
        policy = worker.get_policy()
        policy.set_reward_fn(worker.env.reward_fn)
        return worker, policy

    return setup


def test_reproduce_rewards(setup_worker):
    worker, policy = setup_worker()

    traj = worker.sample()
    tensors = policy._lazy_tensor_dict(traj)
    with torch.no_grad():
        rewards, _ = policy.module.rollout(
            tensors[SampleBatch.ACTIONS],
            tensors[SampleBatch.NEXT_OBS],
            tensors[SampleBatch.CUR_OBS][0],
        )

    target = torch.Tensor(traj[SampleBatch.REWARDS])
    assert torch.allclose(target, rewards, atol=1e-6)


def test_propagates_gradients(setup_worker):
    worker, policy = setup_worker()

    traj = worker.sample()
    tensors = policy._lazy_tensor_dict(traj)
    rewards, _ = policy.module.rollout(
        tensors[SampleBatch.ACTIONS],
        tensors[SampleBatch.NEXT_OBS],
        tensors[SampleBatch.CUR_OBS][0],
    )

    rewards.sum().backward()

    assert all(p.grad is not None for p in policy.module.policy.parameters())
