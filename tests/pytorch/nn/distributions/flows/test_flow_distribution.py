# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

import raylab.pytorch.nn.distributions as ptd
from raylab.pytorch.nn.distributions.flows import Transform
from raylab.pytorch.nn.distributions.flows.utils import sum_rightmost


class Basic1DFlow(Transform):
    def __init__(self, distribution):
        super().__init__(event_dim=1)
        self.distribution = distribution

    def encode(self, inputs):
        out = self.distribution.cdf(inputs)
        log_abs_det_jacobian = self.distribution.log_prob(inputs)
        return out, sum_rightmost(log_abs_det_jacobian, self.event_dim)

    def decode(self, inputs):
        out = self.distribution.icdf(inputs)
        log_abs_det_jacobian = -self.distribution.log_prob(out)
        return out, sum_rightmost(log_abs_det_jacobian, self.event_dim)


@pytest.fixture
def flow():
    return Basic1DFlow(
        ptd.Distribution(
            cond_dist=ptd.Normal(),
            params={"loc": torch.zeros(2), "scale": torch.ones(2)},
        )
    )


@pytest.fixture
def dist(flow, torch_script):
    base_dist = ptd.Distribution(
        cond_dist=ptd.Independent(ptd.Uniform(), reinterpreted_batch_ndims=1),
        params={"low": torch.zeros(2), "high": torch.ones(2)},
    )
    module = ptd.TransformedDistribution(base_dist, flow)
    module = ptd.Distribution(cond_dist=module)
    return torch.jit.script(module) if torch_script else module


def test_basic_flow(flow, torch_script):
    flow = torch.jit.script(flow) if torch_script else flow

    value = torch.randn(10, 2)
    latent, log_det = flow(value)
    assert latent.shape == value.shape
    assert log_det.shape == (10,)

    value_, log_det = flow(latent, reverse=True)
    assert value_.shape == value.shape
    assert torch.allclose(value_, value, atol=1e-7)
    assert log_det.shape == (10,)


def test_dist(dist):
    rsample, logp = dist.rsample((5,))
    assert rsample.dtype == torch.float32
    assert logp.dtype == torch.float32
    assert rsample.shape == (5, 2)
    assert logp.shape == (5,)

    rsample.requires_grad_()
    logp = dist.log_prob(rsample)
    assert logp.dtype == torch.float32
    assert logp.shape == (5,)
