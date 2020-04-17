# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.networks import ResidualNet

PARAMS = (None, {}, {"state": torch.randn(10, 4)})


@pytest.fixture(params=PARAMS, ids=("NoneParams", "EmptyParams", "StateParams"))
def params(request):
    return request.param


def test_creation(params):
    mod = ResidualNet(4, 4, 6)
    mod = torch.jit.script(mod)

    inputs = torch.randn(10, 4)
    out = mod(inputs, params)
    assert out.shape == inputs.shape
