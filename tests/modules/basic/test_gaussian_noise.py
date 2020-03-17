# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.basic import GaussianNoise


@pytest.fixture(params=(pytest.param(-1.0, marks=pytest.mark.xfail), 0, 0.5, 1.0))
def scale(request):
    return request.param


def test_gaussian_noise(scale, torch_script):
    module = GaussianNoise(scale)
    if torch_script:
        module = torch.jit.script(module)

    inputs = torch.randn(10, 4)
    module(inputs)
