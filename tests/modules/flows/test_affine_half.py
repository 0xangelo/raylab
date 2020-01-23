# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn

from raylab.modules import FullyConnected
from raylab.modules.flows import AffineHalfFlow


MLP_KWARGS = {
    "units": (24, 24, 24),
    "activation": {"name": "LeakyReLU", "options": {"negative_slope": 0.2}},
}


def module_fn(kwargs):
    kwargs = kwargs.copy()

    def func(nin, nout):
        nonlocal kwargs
        last = kwargs["units"][-1]
        return nn.Sequential(FullyConnected(nin, **kwargs), nn.Linear(last, nout))

    return func


@pytest.fixture(params=(MLP_KWARGS, None))
def scale_module(request):
    return module_fn(request.param) if request.param else lambda *x: None


@pytest.fixture(params=(MLP_KWARGS, None))
def shift_module(request):
    return module_fn(request.param) if request.param else lambda *x: None


@pytest.fixture(params=(True, False))
def parity(request):
    return request.param


@pytest.fixture
def model(parity, scale_module, shift_module):
    def model_fn(dim):
        nin = dim - (dim // 2)
        nout = dim // 2
        if parity:
            nin, nout = nout, nin
        return AffineHalfFlow(parity, scale_module(nin, nout), shift_module(nin, nout))

    return model_fn


@pytest.fixture(params=(2, 4, 7))
def dim(request):
    return request.param


@pytest.fixture(params=((), (1,), (4,)))
def inputs(request, dim):
    input_shape = request.param + (dim,)
    return torch.randn(*input_shape).requires_grad_()


def test_affine_half(model, inputs):
    model = model(inputs.size(-1))

    model.train()
    latent, log_det = model(inputs)
    if list(model.s_cond.parameters()):
        log_det.sum().backward(retain_graph=True)
        assert all(p.grad is not None for p in model.s_cond.parameters())
    latent.sum().backward()
    assert inputs.grad is not None

    latent = latent.detach().requires_grad_()
    model.eval()
    input_, log_det = model(latent)
    assert torch.allclose(input_, inputs, atol=1e-7)
    if list(model.s_cond.parameters()):
        log_det.sum().backward(retain_graph=True)
        assert all(p.grad is not None for p in model.s_cond.parameters())
    input_.sum().backward()
    assert latent.grad is not None
