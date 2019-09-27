import pytest
import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.utils.debug import fake_batch


@pytest.fixture
def policy_and_batch(svg_policy, obs_space, action_space):
    policy = svg_policy(obs_space, action_space, {"polyak": 0.5})
    batch = policy._lazy_tensor_dict(fake_batch(obs_space, action_space, batch_size=10))
    return policy, batch


def test_target_value_output(policy_and_batch):
    policy, batch = policy_and_batch
    next_vals = policy.module.target_value(batch[SampleBatch.NEXT_OBS])
    assert next_vals.shape == (10, 1)
    assert next_vals.dtype == torch.float32

    targets = policy._compute_value_targets(batch)
    assert targets.shape == (10,)
    assert targets.dtype == torch.float32

    policy.module.zero_grad()
    targets.mean().backward()
    target_params = set(policy.module.target_value.parameters())
    other_params = (p for p in policy.module.parameters() if p not in target_params)
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in other_params)


def test_importance_sampling_weighted_loss(policy_and_batch):
    policy, batch = policy_and_batch
    values = policy.module.value(batch[SampleBatch.CUR_OBS])
    assert values.shape == (10, 1)
    assert values.dtype == torch.float32

    values = values.squeeze(-1)
    targets = torch.randn(10)
    is_ratio = torch.randn(10)
    weighted_losses = nn.MSELoss(reduction="none")(values, targets) * is_ratio
    assert weighted_losses.shape == (10,)

    loss = weighted_losses.div(2).mean()
    loss.backward()
    value_params = set(policy.module.value.parameters())
    other_params = (p for p in policy.module.parameters() if p not in value_params)
    assert all(p.grad is not None for p in value_params)
    assert all(p.grad is None for p in other_params)


def test_target_params_update(policy_and_batch):
    policy, _ = policy_and_batch
    assert all(
        torch.allclose(p, p_)
        for p, p_ in zip(
            policy.module.value.parameters(), policy.module.target_value.parameters()
        )
    )

    old_params = [p.clone() for p in policy.module.target_value.parameters()]
    policy.update_targets()
    assert all(
        not torch.allclose(p, p_)
        for p, p_ in zip(policy.module.target_value.parameters(), old_params)
    )
