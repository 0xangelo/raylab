import pytest
import torch

from raylab.modules.diag_multivariate_normal_rsample import (
    DiagMultivariateNormalRSample,
)


@pytest.fixture(params=(True, False))
def mean_only(request):
    return request.param


@pytest.fixture(params=(True, False))
def squashed(request):
    return request.param


@pytest.fixture(params=((1,), (2,), (4,)))
def action_bounds(request):
    shape = request.param
    action_high = torch.ones(*shape)
    return dict(action_low=action_high.neg(), action_high=action_high)


@pytest.fixture
def module_and_inputs(mean_only, squashed, action_bounds):
    return (
        DiagMultivariateNormalRSample(
            mean_only=mean_only, squashed=squashed, **action_bounds
        ),
        {
            "loc": torch.randn((10,) + action_bounds["action_low"].shape),
            "scale_diag": torch.randn((10,) + action_bounds["action_low"].shape),
        },
    )


def test_forward(module_and_inputs):
    module, inputs = module_and_inputs
    loc = inputs["loc"]
    loc.requires_grad_(True)
    sample, logp = module(inputs)

    assert sample.shape == loc.shape
    assert sample.dtype == torch.float32
    assert logp.shape == (10,)
    assert logp.dtype == torch.float32

    loc.grad = None
    sample.mean().backward()
    assert loc.grad is not None
    assert (loc.grad != 0).any()

    sample, logp = module(inputs)
    loc.grad = None
    logp.mean().backward()
    assert loc.grad is not None
