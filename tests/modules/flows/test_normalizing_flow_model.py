# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn
import torch.distributions as ptd


from raylab.modules.flows import NormalizingFlow, NormalizingFlowModel


class NormalCDF(nn.Module):
    def forward(self, inputs):  # pylint:disable=arguments-differ
        dist = ptd.Normal(torch.zeros(2), torch.ones(2))
        return dist.cdf(inputs), dist.log_prob(inputs).sum(-1)


class NormalICDF(nn.Module):
    def forward(self, inputs):  # pylint:disable=arguments-differ
        dist = ptd.Normal(torch.zeros(2), torch.ones(2))
        value = dist.icdf(inputs)
        # https://math.stackexchange.com/q/910355/595715
        log_det = -dist.log_prob(value).sum(-1)
        return value, log_det


class Basic2DFlow(NormalizingFlow):
    def __init__(self, cdf, icdf):
        super().__init__()
        self.cdf = cdf
        self.icdf = icdf

    def _encode(self, inputs):
        return self.cdf(inputs)

    def _decode(self, inputs):
        return self.icdf(inputs)


@pytest.fixture
def flow():
    cdf = torch.jit.trace(NormalCDF(), torch.randn(1, 2))
    icdf = torch.jit.trace(NormalICDF(), torch.rand(1, 2))
    return Basic2DFlow(cdf, icdf)


@pytest.fixture
def module(flow):
    return NormalizingFlowModel(
        ptd.Independent(
            ptd.Uniform(torch.zeros(2), torch.ones(2)), reinterpreted_batch_ndims=1,
        ),
        [flow],
    )


def test_basic_flow(flow):
    value = torch.randn(10, 2)
    latent, log_det = flow(value)
    assert latent.shape == value.shape
    assert log_det.shape == (10,)

    value_, log_det = flow(latent, reverse=True)
    assert value_.shape == value.shape
    assert torch.allclose(value_, value)
    assert log_det.shape == (10,)


def test_nf_model(module):
    inputs = torch.randn(10, 2, requires_grad=True)

    latent, logp = module(inputs)
    assert (torch.zeros(2) <= latent).all() and (latent <= torch.ones(2)).all()
    assert latent.grad_fn is not None
    assert latent.dtype == torch.float32
    assert logp.dtype == torch.float32
    assert logp.shape == (10,)

    sample, logp = module.rsample(5)
    assert sample.dtype == torch.float32
    assert logp.dtype == torch.float32
    assert sample.shape == (5, 2)
    assert logp.shape == (5,)
