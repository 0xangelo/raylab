# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn

from raylab.modules.flows import AffineConstantFlow, ActNorm


@pytest.fixture(params=(True, False))
def scale(request):
    return request.param


@pytest.fixture(params=(True, False))
def shift(request):
    return request.param


@pytest.fixture(params=(AffineConstantFlow, ActNorm))
def model(request, scale, shift):
    return lambda dim: request.param(dim, scale, shift)


@pytest.fixture(params=(1, 2, 4))
def dim(request):
    return request.param


@pytest.fixture(params=((), (1,), (4,)))
def inputs(request, dim):
    input_shape = request.param + (dim,)
    return torch.randn(*input_shape).requires_grad_()


def test_affine_constant(model, inputs):
    model = model(inputs.size(-1))

    model.train()
    latent, log_det = model(inputs)
    if isinstance(model.scale, nn.Parameter):
        log_det.sum().backward(retain_graph=True)
        assert model.scale.grad is not None
    latent.sum().backward()
    assert inputs.grad is not None
    assert latent.grad is None

    latent = latent.detach().requires_grad_()
    model.eval()
    input_, log_det = model(latent)
    assert torch.allclose(input_, inputs, atol=1e-6)
    if isinstance(model.scale, nn.Parameter):
        log_det.sum().backward(retain_graph=True)
        assert model.scale.grad is not None
    input_.sum().backward()
    assert latent.grad is not None
