# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import torch
import pytest

from raylab.modules.distributions import Independent, Uniform


@pytest.fixture
def uniform(torch_script):
    return torch.jit.script(Uniform()) if torch_script else Uniform()


@pytest.fixture
def independent_uniform(torch_script):
    dist = Independent(Uniform(), reinterpreted_batch_ndims=1)
    return torch.jit.script(dist) if torch_script else dist


@pytest.fixture(params=((), (1,), (2,)))
def sample_shape(request):
    return request.param


def test_uniform(uniform, sample_shape):
    dist = uniform
    flat = torch.stack([torch.zeros([]), torch.ones([])], dim=0)

    params = dist(flat)
    assert "low" in params
    assert "high" in params
    event_shape = params["low"].shape

    sample = dist.sample(params, sample_shape)
    rsample = dist.rsample(params, sample_shape)
    assert sample.shape == sample_shape + event_shape
    assert rsample.shape == sample_shape + event_shape

    log_prob = dist.log_prob(params, sample)
    assert log_prob.shape == sample_shape + event_shape
    cdf = dist.cdf(params, sample)
    assert cdf.shape == sample_shape + event_shape
    icdf = dist.icdf(params, cdf)
    assert icdf.shape == sample.shape
    entropy = dist.entropy(params)
    assert entropy.shape == event_shape
    perplexity = dist.perplexity(params)
    assert perplexity.shape == entropy.shape


def test_independent_uniform(independent_uniform, sample_shape):
    dist = independent_uniform
    flat = torch.cat([torch.zeros(2), torch.ones(2)], dim=0)

    params = dist(flat)
    assert "low" in params
    assert "high" in params
    assert params["low"].shape == (2,)
    assert params["high"].shape == (2,)
    batch_shape = params["low"].shape[:-1]
    event_shape = params["low"].shape[-1:]

    sample = dist.sample(params, sample_shape)
    rsample = dist.rsample(params, sample_shape)
    assert sample.shape == sample_shape + batch_shape + event_shape
    assert rsample.shape == sample_shape + batch_shape + event_shape

    log_prob = dist.log_prob(params, sample)
    assert log_prob.shape == sample_shape + batch_shape
    cdf = dist.cdf(params, sample)
    assert cdf.shape == sample_shape + batch_shape
    entropy = dist.entropy(params)
    assert entropy.shape == batch_shape
    perplexity = dist.perplexity(params)
    assert perplexity.shape == entropy.shape
