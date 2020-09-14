import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture
def policy(policy_fn, svg_inf_policy):
    policy = policy_fn(svg_inf_policy, {})
    policy.set_reward_from_config()
    return policy


@pytest.fixture
def batch(policy, batch_fn):
    return batch_fn(policy)


def test_reproduce_rewards(policy, batch):
    with torch.no_grad():
        rewards = policy.loss_actor._rollout(
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.NEXT_OBS],
            batch[SampleBatch.CUR_OBS][0],
        )

    assert torch.allclose(batch[SampleBatch.REWARDS], rewards, atol=1e-6)


def test_propagates_gradients(policy, batch):
    rewards = policy.loss_actor._rollout(
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
        batch[SampleBatch.CUR_OBS][0],
    )

    rewards.sum().backward()

    assert all(p.grad is not None for p in policy.module.actor.parameters())
