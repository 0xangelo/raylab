# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch


GRAD_ESTIMATOR = "SF PD".split()


@pytest.fixture(params=GRAD_ESTIMATOR)
def grad_estimator(request):
    return request.param


@pytest.fixture(params=(1, 2, 4))
def num_model_samples(request):
    return request.param


@pytest.fixture
def policy_and_batch(policy_and_batch_fn):
    return policy_and_batch_fn({})


def test_score_function(policy_and_batch_fn, grad_estimator, num_model_samples):
    policy, batch = policy_and_batch_fn(
        {"grad_estimator": grad_estimator, "num_model_samples": num_model_samples}
    )

    obs = batch[SampleBatch.CUR_OBS]
    acts = batch[SampleBatch.ACTIONS].clone().requires_grad_()

    sample, logp = policy.module.model.sample(obs, acts)
    assert sample.grad_fn is None
    logp.mean().backward()
    assert acts.grad is not None
    assert not torch.allclose(acts.grad, torch.zeros_like(acts))


def test_daml_loss(policy_and_batch_fn, grad_estimator, num_model_samples):
    policy, batch = policy_and_batch_fn(
        {"grad_estimator": grad_estimator, "num_model_samples": num_model_samples}
    )

    loss, info = policy.daml_loss(batch, policy.module, policy.config)
    assert isinstance(info, dict)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    policy.module.zero_grad()
    loss.backward()
    params = list(policy.module.model.parameters())
    assert all(p.grad is not None for p in params)
    assert all(torch.isfinite(p.grad).all() for p in params)
    assert all(not torch.isnan(p.grad).any() for p in params)
    assert all(not torch.allclose(p.grad, torch.zeros_like(p)) for p in params)


def test_mle_loss(policy_and_batch):
    policy, batch = policy_and_batch

    loss, info = policy.mle_loss(batch, policy.module)
    assert isinstance(info, dict)
    assert loss.shape == ()
    assert loss.dtype == torch.float32

    policy.module.zero_grad()
    loss.backward()
    params = list(policy.module.model.parameters())
    assert all(p.grad is not None for p in params)
    assert all(torch.isfinite(p.grad).all() for p in params)
    assert all(not torch.isnan(p.grad).any() for p in params)
    assert all(not torch.allclose(p.grad, torch.zeros_like(p)) for p in params)
