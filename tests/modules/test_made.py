# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


from raylab.modules import MADE


@pytest.fixture(params=(True, False), ids=("NaturalOrder", "RandomOrder"))
def model(request):
    return MADE(3, (4, 4), 3, natural_ordering=request.param)


@pytest.fixture
def nat_order_model():
    return MADE(3, (4, 4), 3, natural_ordering=True)


def test_made_creation(model, torch_script):
    model = torch.jit.script(model) if torch_script else model


def test_made_forward(nat_order_model):
    model = nat_order_model

    inputs = torch.randn(3).requires_grad_()
    out = model(inputs)

    for idx in range(3):
        out[..., idx].sum().backward(retain_graph=True)
        assert (inputs.grad[..., idx:] == 0).all().item()
