# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.pytorch.nn import TanhSquash


@pytest.fixture(params=(1, 2, 4))
def low_high(request):
    high = torch.randn(request.param).abs()
    low = high.neg()
    return low, high


@pytest.fixture
def squash(low_high):
    low, high = low_high
    return TanhSquash(low, high)


@pytest.fixture
def module(squash, torch_script):
    if torch_script:
        return torch.jit.script(squash)
    return squash


@pytest.fixture
def inputs(low_high):
    low, _ = low_high
    return torch.randn(10, *low.shape)


def test_squash_to_range(module, low_high, inputs):
    low, high = low_high

    output = module(inputs)
    assert (output <= high).all()
    assert (output >= low).all()


def test_propagates_gradients(module, inputs):
    inputs.requires_grad_()
    module(inputs).mean().backward()
    assert inputs.grad is not None
    assert (inputs.grad != 0).any()


def test_reverse(module, inputs):
    squashed = module(inputs)
    assert torch.allclose(module(squashed, reverse=True), inputs, atol=1e-6)
