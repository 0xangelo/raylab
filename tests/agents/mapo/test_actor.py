# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


GRAD_ESTIMATOR = "SF PD".split()


@pytest.fixture(params=GRAD_ESTIMATOR)
def grad_estimator(request):
    return request.param


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, grad_estimator):
    return policy_and_batch_fn({"grad_estimator": grad_estimator})


def test_policy_loss(policy_and_batch):
    policy, batch = policy_and_batch

    loss, info = policy.madpg_loss(batch, policy.module, policy.config)
    assert isinstance(info, dict)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    policy.module.zero_grad()
    loss.backward()
    params = list(policy.module.actor.parameters())
    assert all(p.grad is not None for p in params)
    assert all(torch.isfinite(p.grad).all() for p in params)
    assert all(not torch.isnan(p.grad).any() for p in params)
    assert all(not torch.allclose(p.grad, torch.zeros_like(p)) for p in params)
