# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture(scope="module")
def base_cls():
    from raylab.policy.modules.actor.policy.stochastic import MLPStochasticPolicy

    return MLPStochasticPolicy


@pytest.fixture(scope="module")
def cont_cls():
    from raylab.policy.modules.actor.policy.stochastic import MLPContinuousPolicy

    return MLPContinuousPolicy


@pytest.fixture(scope="module")
def disc_cls():
    from raylab.policy.modules.actor.policy.stochastic import MLPDiscretePolicy

    return MLPDiscretePolicy


@pytest.fixture
def spec(base_cls):
    return base_cls.spec_cls()


@pytest.fixture(params=(True, False), ids=lambda x: f"InputDependentScale({x})")
def input_dependent_scale(request):
    return request.param


@pytest.fixture
def cont_policy(cont_cls, obs_space, cont_space, spec, input_dependent_scale):
    return cont_cls(obs_space, cont_space, spec, input_dependent_scale)


@pytest.fixture
def disc_policy(disc_cls, obs_space, disc_space, spec):
    return disc_cls(obs_space, disc_space, spec)


def test_continuous_sample(cont_policy, cont_batch):
    policy, batch = cont_policy, cont_batch
    action = batch[SampleBatch.ACTIONS]

    sampler = policy.rsample
    samples, logp = sampler(batch[SampleBatch.CUR_OBS])
    samples_, _ = sampler(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert not torch.allclose(samples, samples_)


def test_discrete_sample(disc_policy, disc_batch):
    policy, batch = disc_policy, disc_batch
    action = batch[SampleBatch.ACTIONS]

    sampler = policy.sample
    samples, logp = sampler(batch[SampleBatch.CUR_OBS])
    samples_, _ = sampler(batch[SampleBatch.CUR_OBS])
    assert samples.shape == action.shape
    assert samples.dtype == action.dtype
    assert logp.shape == batch[SampleBatch.REWARDS].shape
    assert logp.dtype == batch[SampleBatch.REWARDS].dtype
    assert not torch.allclose(samples, samples_)


def test_continuous_params(cont_policy, cont_batch):
    policy, batch = cont_policy, cont_batch
    params = policy(batch[SampleBatch.CUR_OBS])
    assert "loc" in params
    assert "scale" in params

    loc, scale = params["loc"], params["scale"]
    action = batch[SampleBatch.ACTIONS]
    assert loc.shape == action.shape
    assert scale.shape == action.shape
    assert loc.dtype == torch.float32
    assert scale.dtype == torch.float32

    pi_params = set(policy.parameters())
    for par in pi_params:
        par.grad = None
    loc.mean().backward()
    assert any(p.grad is not None for p in pi_params)

    for par in pi_params:
        par.grad = None
    policy(batch[SampleBatch.CUR_OBS])["scale"].mean().backward()
    assert any(p.grad is not None for p in pi_params)


def test_discrete_params(disc_policy, disc_space, disc_batch):
    policy, batch = disc_policy, disc_batch

    params = policy(batch[SampleBatch.CUR_OBS])
    assert "logits" in params
    logits = params["logits"]
    assert logits.shape[-1] == disc_space.n

    pi_params = set(policy.parameters())
    for par in pi_params:
        par.grad = None
    logits.mean().backward()
    assert any(p.grad is not None for p in pi_params)


def test_reproduce(cont_policy, cont_batch):
    policy, batch = cont_policy, cont_batch

    acts = batch[SampleBatch.ACTIONS]
    acts_, logp_ = policy.reproduce(batch[SampleBatch.CUR_OBS], acts)
    assert acts_.shape == acts.shape
    assert acts_.dtype == acts.dtype
    assert torch.allclose(acts_, acts, atol=1e-5)
    assert logp_.shape == batch[SampleBatch.REWARDS].shape

    acts_.mean().backward()
    pi_params = set(policy.parameters())
    assert all(p.grad is not None for p in pi_params)
