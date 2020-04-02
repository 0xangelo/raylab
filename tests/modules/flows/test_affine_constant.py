# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn

from raylab.modules.flows import AffineConstantFlow, ActNorm


@pytest.fixture(params=(True, False), ids=("LearnScale", "ConstScale"))
def scale(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("LearnShift", "ConstShift"))
def shift(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("AffineConst", "ActNorm"))
def module(request, scale, shift):
    def make_mod(shape):
        mod = AffineConstantFlow(shape, scale, shift)
        return ActNorm(mod) if request.param else mod

    return make_mod


@pytest.fixture(params=((1,), (2,), (4,)))
def shape(request):
    return request.param


@pytest.fixture(params=((), (1,), (4,)))
def inputs(request, shape):
    input_shape = request.param + shape
    return torch.randn(*input_shape).requires_grad_()


def test_affine_constant(module, inputs, torch_script):
    module = module(inputs.shape[-1:])
    module = torch.jit.script(module) if torch_script else module
    scale = module.scale if "scale" in dir(module) else module.affine_const.scale

    latent, log_det = module(inputs)
    if isinstance(scale, nn.Parameter):
        log_det.sum().backward(retain_graph=True)
        assert scale.grad is not None
    latent.sum().backward()
    assert inputs.grad is not None

    latent = latent.detach().requires_grad_()
    input_, log_det = module(latent, reverse=True)
    assert torch.allclose(input_, inputs, atol=1e-6)
    if isinstance(scale, nn.Parameter):
        log_det.sum().backward(retain_graph=True)
        assert scale.grad is not None
    input_.sum().backward()
    assert latent.grad is not None
