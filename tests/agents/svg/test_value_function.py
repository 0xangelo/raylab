# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn
from ray.rllib import SampleBatch


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, svg_policy):
    return policy_and_batch_fn(svg_policy, {"polyak": 0.5})


def test_compute_value_targets(policy_and_batch):
    policy, batch = policy_and_batch

    targets = policy._compute_value_targets(batch)
    assert targets.shape == (10,)
    assert targets.dtype == torch.float32
    assert torch.allclose(
        targets[batch[SampleBatch.DONES]],
        batch[SampleBatch.REWARDS][batch[SampleBatch.DONES]],
    )

    policy.module.zero_grad()
    targets.mean().backward()
    target_params = set(policy.module.target_critic.parameters())
    other_params = (p for p in policy.module.parameters() if p not in target_params)
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in other_params)


def test_importance_sampling_weighted_loss(policy_and_batch):
    policy, batch = policy_and_batch

    values = policy.module.critic(batch[SampleBatch.CUR_OBS])
    values = values.squeeze(-1)
    targets = torch.randn(10)
    is_ratio = torch.randn(10)
    weighted_losses = nn.MSELoss(reduction="none")(values, targets) * is_ratio
    assert weighted_losses.shape == (10,)

    loss = weighted_losses.div(2).mean()
    loss.backward()
    value_params = set(policy.module.critic.parameters())
    other_params = (p for p in policy.module.parameters() if p not in value_params)
    assert all(p.grad is not None for p in value_params)
    assert all(p.grad is None for p in other_params)


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
