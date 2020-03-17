# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.basic import TanhSquash


@pytest.fixture(params=(1, 2, 4))
def low_high(request):
    high = torch.randn(request.param).abs()
    low = high.neg()
    return low, high


@pytest.fixture
def maker(torch_script):
    def factory(*args, **kwargs):
        module = TanhSquash(*args, **kwargs)
        return torch.jit.script(module) if torch_script else module

    return factory


def test_squash_to_range(maker, low_high):
    low, high = low_high
    module = maker(low, high)

    inputs = torch.randn(10, *low.shape)
    output = module(inputs)
    assert (output <= high).all()
    assert (output >= low).all()


def test_propagates_gradients(maker, low_high):
    low, high = low_high
    module = maker(low, high)

    inputs = torch.randn(10, *low.shape, requires_grad=True)
    module(inputs).mean().backward()
    assert inputs.grad is not None
    assert (inputs.grad != 0).any()
