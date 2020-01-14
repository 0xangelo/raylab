# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch


@pytest.fixture
def config():
    return {}


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, config):
    return policy_and_batch_fn(config)


@pytest.fixture(params=(1, 2, 4))
def num_model_samples(request):
    return request.param


@pytest.fixture(params=(1, 5, 10))
def model_rollout_len(request):
    return request.param


def test_model_output(policy_and_batch, num_model_samples):
    policy, batch = policy_and_batch

    next_obs, logp = policy.module.model_sampler(
        batch[SampleBatch.CUR_OBS],
        batch[SampleBatch.ACTIONS],
        torch.as_tensor([num_model_samples]),
    )
    assert next_obs.shape == (num_model_samples,) + batch[SampleBatch.NEXT_OBS].shape
    assert next_obs.dtype == torch.float32
    assert not torch.isnan(next_obs).any()
    assert torch.isfinite(next_obs).all()
    assert logp.shape == (num_model_samples,) + batch[SampleBatch.REWARDS].shape
    assert logp.dtype == torch.float32
    assert not torch.isnan(logp).any()
    assert torch.isfinite(logp).all()


def test_model_logp(policy_and_batch):
    policy, batch = policy_and_batch

    logp = policy.module.model_logp(
        batch[SampleBatch.CUR_OBS],
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
    )
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == torch.float32
    assert not torch.isnan(logp).any()
    assert torch.isfinite(logp).all()


def test_madpg_loss(policy_and_batch, num_model_samples, model_rollout_len):
    policy, batch = policy_and_batch
    policy.config["num_model_samples"] = num_model_samples
    policy.config["model_rollout_len"] = model_rollout_len

    loss, info = policy.compute_decision_aware_loss(batch, policy.module, policy.config)
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


def test_decision_aware_loss(policy_and_batch):
    policy, batch = policy_and_batch

    loss, info = policy.compute_decision_aware_loss(batch, policy.module, policy.config)
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
