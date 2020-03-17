# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.basic import NormalizedLinear


@pytest.fixture(params=(1, 2, 4))
def dim(request):
    return request.param


@pytest.fixture
def input_dim(dim):
    return dim


@pytest.fixture
def output_dim(dim):
    return dim


@pytest.fixture(params=(0.8, 1.0, 1.2))
def beta(request):
    return request.param


def test_normalizes_vector(input_dim, output_dim, beta, torch_script):
    module = NormalizedLinear(input_dim, output_dim, beta=beta)
    if torch_script:
        module = torch.jit.script(module)

    inputs = torch.randn(10, input_dim)
    output = module(inputs)
    norms = output.norm(p=1, dim=-1, keepdim=True) / output_dim
    atol = torch.finfo(torch.float32).eps
    assert (norms <= (beta + atol)).all()


def test_propagates_gradients(input_dim, output_dim, beta, torch_script):
    module = NormalizedLinear(input_dim, output_dim, beta=beta)
    if torch_script:
        module = torch.jit.script(module)

    inputs = torch.randn(10, input_dim, requires_grad=True)
    module(inputs).mean().backward()
    assert inputs.grad is not None
    assert (inputs.grad != 0).any()
