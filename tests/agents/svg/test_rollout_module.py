# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, svg_inf_policy):
    policy, batch = policy_and_batch_fn(svg_inf_policy, {})
    policy.set_reward_from_config(policy.config["env"], policy.config["env_config"])
    return policy, batch


def test_reproduce_rewards(policy_and_batch):
    policy, batch = policy_and_batch

    with torch.no_grad():
        rewards = policy.loss_actor._rollout(
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.NEXT_OBS],
            batch[SampleBatch.CUR_OBS][0],
        )

    assert torch.allclose(batch[SampleBatch.REWARDS], rewards, atol=1e-6)


def test_propagates_gradients(policy_and_batch):
    policy, batch = policy_and_batch

    rewards = policy.loss_actor._rollout(
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
        batch[SampleBatch.CUR_OBS][0],
    )

    rewards.sum().backward()

    assert all(p.grad is not None for p in policy.module.actor.parameters())
