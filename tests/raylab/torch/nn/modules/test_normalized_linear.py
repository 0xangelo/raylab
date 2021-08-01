import pytest
import torch
from torch import Tensor, nn

from raylab.torch.nn import NormalizedLinear


@pytest.fixture(params=(1, 2))
def dim(request):
    return request.param


@pytest.fixture
def input_dim(dim):
    return dim


@pytest.fixture
def output_dim(dim):
    return dim


@pytest.fixture(params=(0.8, 1.2))
def beta(request):
    return request.param


@pytest.fixture
def inputs(input_dim, beta):
    return torch.rand(10, input_dim) + torch.full((10, input_dim), beta)


@pytest.fixture
def module(input_dim, output_dim, beta, torch_script):
    module = NormalizedLinear(input_dim, output_dim, beta=beta)
    if torch_script:
        module = torch.jit.script(module)
    return module


def test_normalizes_vector(module, inputs, output_dim, beta):
    output = module(inputs)
    norms = output.norm(p=1, dim=-1, keepdim=True) / output_dim
    atol = 1e-6
    assert (norms <= (beta + atol)).all()


@pytest.fixture
def unclipped_inputs(input_dim: int, beta: float):
    # Try to force input norm to less than beta
    return torch.full(size=(10, input_dim), fill_value=beta / (input_dim + 1))


def test_unclipped_values_gradient_propagation(
    module: NormalizedLinear, unclipped_inputs: Tensor
):
    # Force underlying linear module to apply identity map
    nn.init.eye_(module.linear.weight)
    nn.init.zeros_(module.linear.bias)

    inputs = unclipped_inputs.requires_grad_(True)
    module(inputs).mean().backward()
    assert inputs.grad is not None
    assert (inputs.grad != 0).any()
