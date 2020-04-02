# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn

from raylab.modules import FullyConnected
from raylab.modules.flows import Affine1DHalfFlow


MLP_KWARGS = {
    "units": (24,) * 3,
    "activation": {"name": "LeakyReLU", "options": {"negative_slope": 0.2}},
}


def module_fn(kwargs):
    kwargs = kwargs.copy()

    def func(nin, nout):
        nonlocal kwargs
        encoder = FullyConnected(nin, **kwargs)
        return nn.Sequential(encoder, nn.Linear(encoder.out_features, nout))

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
def module(parity, scale_module, shift_module, torch_script):
    def module_fn(size):
        nin = size - (size // 2)
        nout = size // 2
        if parity:
            nin, nout = nout, nin
        scale, shift = scale_module(nin, nout), shift_module(nin, nout)
        mod = Affine1DHalfFlow(parity, scale, shift)
        return torch.jit.script(mod) if torch_script else mod

    return module_fn


@pytest.fixture(params=(2, 4, 7))
def size(request):
    return request.param


@pytest.fixture(params=((), (1,), (4,)))
def inputs(request, size):
    input_shape = request.param + (size,)
    return torch.randn(*input_shape).requires_grad_()


def test_affine_half(module, inputs):
    module = module(inputs.size(-1))

    latent, log_det = module(inputs)
    if list(module.scale.parameters()):
        log_det.sum().backward(retain_graph=True)
        assert all(p.grad is not None for p in module.scale.parameters())
    latent.sum().backward()
    assert inputs.grad is not None

    latent = latent.detach().requires_grad_()

    input_, log_det = module(latent, reverse=True)
    assert torch.allclose(input_, inputs, atol=1e-7)
    if list(module.scale.parameters()):
        log_det.sum().backward(retain_graph=True)
        assert all(p.grad is not None for p in module.scale.parameters())
    input_.sum().backward()
    assert latent.grad is not None
