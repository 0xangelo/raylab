import pytest
import torch

from raylab.modules import DistRSample, DistMean
from raylab.distributions import DiagMultivariateNormal


@pytest.fixture(params=(DistRSample, DistMean))
def module_cls(request):
    return request.param


@pytest.fixture
def dist_cls():
    return DiagMultivariateNormal


@pytest.fixture(params=(True, False))
def squash(request):
    return request.param


@pytest.fixture(params=((1,), (2,), (4,)))
def action_bounds(request):
    shape = request.param
    action_high = torch.ones(*shape)
    return dict(low=action_high.neg(), high=action_high)


@pytest.fixture
def module_and_inputs_fn(dist_cls, action_bounds, squash):
    bounds = action_bounds if squash else dict(low=None, high=None)

    def func(module_cls):
        return (
            module_cls(dist_cls, **bounds),
            {
                "loc": torch.randn((10,) + action_bounds["low"].shape),
                "scale_diag": torch.randn((10,) + action_bounds["low"].shape),
            },
        )

    return func


@pytest.fixture
def module_and_inputs(module_and_inputs_fn, module_cls):
    return module_and_inputs_fn(module_cls)


@pytest.fixture
def mean_module_and_inputs(module_and_inputs_fn):
    return module_and_inputs_fn(DistMean)


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


def test_mean_only_is_deterministic(mean_module_and_inputs):
    module, inputs = mean_module_and_inputs
    loc, scale = inputs["loc"], inputs["scale_diag"]
    loc.requires_grad_(True)
    scale.requires_grad_(True)

    var1, _ = module(inputs)
    var2, _ = module(inputs)
    assert torch.allclose(var1, var2)

    (var1 + var2).mean().backward()
    assert loc.grad is not None
    assert scale.grad is None
