import pytest
import torch
import torch.nn as nn
from ray.rllib.policy.sample_batch import SampleBatch


@pytest.fixture(params=(True, False))
def sampler_noise(request):
    return request.param


@pytest.fixture(params=(0.8, 1.0, 1.2))
def beta(request):
    return request.param


@pytest.fixture
def config(sampler_noise, beta):
    return {"sampler_noise": sampler_noise, "beta": beta}


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, config):
    return policy_and_batch_fn(config)


def test_policy_output(policy_and_batch):
    policy, batch = policy_and_batch

    policy_out = policy.module.policy(batch[SampleBatch.CUR_OBS])
    norms = policy_out.norm(p=1, dim=-1, keepdim=True) / policy.action_space.shape[0]
    assert policy_out.shape[-1] == policy.action_space.shape[0]
    assert policy_out.dtype == torch.float32
    assert (norms <= (policy.config["beta"] + torch.finfo(torch.float32).eps)).all()


def test_policy_sample(policy_and_batch):
    policy, batch = policy_and_batch

    samples = policy.module.sampler(batch[SampleBatch.CUR_OBS])
    samples_ = policy.module.sampler(batch[SampleBatch.CUR_OBS])
    assert samples.shape[-1] == policy.action_space.shape[0]
    assert samples.dtype == torch.float32
    assert not (policy.config["sampler_noise"] and torch.allclose(samples, samples_))
