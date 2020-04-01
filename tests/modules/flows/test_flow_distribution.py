# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


from raylab.modules.flows import NormalizingFlow
from raylab.modules.distributions import (
    Distribution,
    Independent,
    Normal,
    TransformedDistribution,
    Uniform,
)


# class NormalCDF(nn.Module):
#     def forward(self, inputs):  # pylint:disable=arguments-differ
#         dist = ptd.Normal(torch.zeros(2), torch.ones(2))
#         return dist.cdf(inputs), dist.log_prob(inputs).sum(-1)


# class NormalICDF(nn.Module):
#     def forward(self, inputs):  # pylint:disable=arguments-differ
#         dist = ptd.Normal(torch.zeros(2), torch.ones(2))
#         value = dist.icdf(inputs)
#         # https://math.stackexchange.com/q/910355/595715
#         log_det = -dist.log_prob(value).sum(-1)
#         return value, log_det


class Basic2DFlow(NormalizingFlow):
    def __init__(self, distribution):
        super().__init__()
        self.distribution = distribution

    def _encode(self, inputs):
        out = self.distribution.cdf(inputs)
        log_abs_det_jacobian = self.distribution.log_prob(inputs)
        return out, log_abs_det_jacobian

    def _decode(self, inputs):
        out = self.distribution.icdf(inputs)
        log_abs_det_jacobian = -self.distribution.log_prob(out)
        return out, log_abs_det_jacobian


@pytest.fixture
def flow():
    return Basic2DFlow(
        Distribution(
            cond_dist=Independent(Normal(), reinterpreted_batch_ndims=1),
            params={"loc": torch.zeros(2), "scale": torch.ones(2)},
        )
    )


@pytest.fixture
def dist(flow, torch_script):
    base_dist = Distribution(
        cond_dist=Independent(Uniform(), reinterpreted_batch_ndims=1),
        params={"low": torch.zeros(2), "high": torch.ones(2)},
    )
    module = TransformedDistribution(base_dist, flow)
    module = Distribution(cond_dist=module)
    return torch.jit.script(module) if torch_script else module


def test_basic_flow(flow, torch_script):
    flow = torch.jit.script(flow) if torch_script else flow

    value = torch.randn(10, 2)
    latent, log_det = flow(value)
    assert latent.shape == value.shape
    assert log_det.shape == (10,)

    value_, log_det = flow(latent, reverse=True)
    assert value_.shape == value.shape
    assert torch.allclose(value_, value)
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
