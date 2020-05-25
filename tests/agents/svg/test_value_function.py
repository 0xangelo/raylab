# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch

from raylab.utils.dictionaries import get_keys


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, svg_policy):
    return policy_and_batch_fn(svg_policy, {"polyak": 0.5})


def test_compute_value_targets(policy_and_batch):
    policy, batch = policy_and_batch

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


def test_importance_sampling_weighted_loss(policy_and_batch):
    policy, batch = policy_and_batch
    batch[policy.loss_critic.IS_RATIOS] = torch.randn_like(batch[SampleBatch.REWARDS])

    loss, info = policy.loss_critic(batch)
    loss.backward()
    value_params = set(policy.module.critic.parameters())
    other_params = (p for p in policy.module.parameters() if p not in value_params)
    assert all(p.grad is not None for p in value_params)
    assert all(p.grad is None for p in other_params)

    assert "loss(critic)" in info


def test_target_params_update(policy_and_batch):
    policy, _ = policy_and_batch

    old_params = [p.clone() for p in policy.module.target_critic.parameters()]
    for param in policy.module.critic.parameters():
        param.data.add_(torch.ones_like(param))
    policy.update_targets("critic", "target_critic")
    assert all(
        not torch.allclose(p, p_)
        for p, p_ in zip(policy.module.target_critic.parameters(), old_params)
    )
