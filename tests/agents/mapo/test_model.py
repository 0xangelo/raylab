# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


@pytest.fixture
def policy_and_batch(policy_and_batch_fn):
    return policy_and_batch_fn({})


@pytest.fixture(params=(1, 2, 4))
def num_model_samples(request):
    return request.param


@pytest.fixture(params=(1, 5, 10))
def model_rollout_len(request):
    return request.param


def test_madpg_loss(policy_and_batch, num_model_samples, model_rollout_len):
    policy, batch = policy_and_batch
    policy.config["num_model_samples"] = num_model_samples
    policy.config["model_rollout_len"] = model_rollout_len

    loss, info = policy.compute_daml_loss(batch, policy.module, policy.config)
    assert isinstance(info, dict)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    policy.module.zero_grad()
    loss.backward()
    assert all(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and not torch.isnan(p.grad).all()
        for p in policy.module.model.parameters()
    )


def test_daml_loss(policy_and_batch):
    policy, batch = policy_and_batch

    loss, info = policy.compute_daml_loss(batch, policy.module, policy.config)
    assert isinstance(info, dict)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    policy.module.zero_grad()
    loss.backward()
    assert all(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and not torch.isnan(p.grad).all()
        for p in policy.module.model.parameters()
    )


def test_mle_loss(policy_and_batch):
    policy, batch = policy_and_batch

    loss, info = policy.compute_mle_loss(batch, policy.module)
    assert isinstance(info, dict)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    policy.module.zero_grad()
    loss.backward()
    assert all(
        p.grad is not None
        and torch.isfinite(p.grad).all()
        and not torch.isnan(p.grad).all()
        for p in policy.module.model.parameters()
    )
