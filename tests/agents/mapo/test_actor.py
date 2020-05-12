# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


@pytest.fixture
def policy_and_batch(policy_and_batch_fn):
    return policy_and_batch_fn({})


def test_policy_loss(policy_and_batch):
    policy, batch = policy_and_batch

    loss, info = policy.madpg_loss(batch, policy.module, policy.config)
    assert isinstance(info, dict)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    policy.module.zero_grad()
    loss.backward()
    assert all(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and not torch.isnan(p.grad).any()
        for p in policy.module.actor.parameters()
    )
