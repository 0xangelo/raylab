# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules import TanhSquash


@pytest.fixture(params=(1, 2, 4))
def input_dim(request):
    return request.param


def test_squash_to_range(input_dim):
    high = torch.randn(input_dim).abs()
    low = high.neg()
    module = TanhSquash(low, high)

    inputs = torch.randn(10, input_dim)
    output = module(inputs)
    assert (output <= high).all()
    assert (output >= low).all()


def test_propagates_gradients(input_dim):
    high = torch.randn(input_dim).abs()
    low = high.neg()
    module = TanhSquash(low, high)

    inputs = torch.randn(10, input_dim, requires_grad=True)
    module(inputs).mean().backward()
    assert inputs.grad is not None
    assert (inputs.grad != 0).any()
