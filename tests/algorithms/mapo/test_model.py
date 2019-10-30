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


def test_model_output(policy_and_batch):
    policy, batch = policy_and_batch

    next_obs, logp = policy.module.model_sampler(
        batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
    )
    assert next_obs.shape == batch[SampleBatch.NEXT_OBS].shape
    assert next_obs.dtype == torch.float32
    assert not torch.isnan(next_obs).any()
    assert torch.isfinite(next_obs).all()
    assert logp.shape == batch[SampleBatch.REWARDS].shape
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
