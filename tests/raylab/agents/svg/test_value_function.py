import pytest
import torch
from ray.rllib import SampleBatch

from raylab.utils.dictionaries import get_keys


@pytest.fixture
def policy(policy_fn, svg_policy):
    policy = policy_fn(svg_policy, {})
    policy.set_reward_from_config()
    return policy


@pytest.fixture
def batch(policy, batch_fn):
    return batch_fn(policy)


def test_compute_value_targets(policy, batch):
    rewards, dones = get_keys(batch, SampleBatch.REWARDS, SampleBatch.DONES)
    targets = policy.loss_critic.sampled_one_step_state_values(batch)
    assert targets.shape == (10,)
    assert targets.dtype == torch.float32
    assert torch.allclose(targets[dones], rewards[dones])

    policy.module.zero_grad()
    targets.mean().backward()
    target_params = set(policy.module.target_critic.parameters())
    other_params = (p for p in policy.module.parameters() if p not in target_params)
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in other_params)


def test_importance_sampling_weighted_loss(policy, batch):
    batch[policy.loss_critic.IS_RATIOS] = torch.randn_like(batch[SampleBatch.REWARDS])

    loss, info = policy.loss_critic(batch)
    loss.backward()
    value_params = set(policy.module.critic.parameters())
    other_params = (p for p in policy.module.parameters() if p not in value_params)
    assert all(p.grad is not None for p in value_params)
    assert all(p.grad is None for p in other_params)

    assert "loss(critic)" in info
