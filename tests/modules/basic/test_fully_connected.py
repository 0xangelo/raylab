# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.basic import FullyConnected


@pytest.fixture(params=(True, False))
def torch_script(request):
    return request.param


@pytest.fixture(params=(1, 2, 4))
def in_features(request):
    return request.param


@pytest.fixture(params=((), (10,), (4, 4)))
def units(request):
    return request.param


@pytest.fixture(params=(None, "Tanh", "ReLU", "ELU"))
def activation(request):
    return request.param


@pytest.fixture(params=(True, False))
def layer_norm(request):
    return request.param


@pytest.fixture(params=("xavier_uniform", "orthogonal"))
def initializer_options(request):
    return {"name": request.param}


@pytest.fixture
def kwargs(in_features, units, activation, layer_norm, initializer_options):
    return dict(
        in_features=in_features,
        units=units,
        activation=activation,
        layer_norm=layer_norm,
        **initializer_options
    )


def test_fully_connected(kwargs, torch_script):
    maker = FullyConnected.as_script_module if torch_script else FullyConnected
    module = maker(**kwargs)

    inputs = torch.randn(1, kwargs["in_features"])
    module(inputs)
