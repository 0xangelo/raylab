# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch

from raylab.modules.basic import FullyConnected, StateActionEncoder


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
def kwargs(units, activation, layer_norm, initializer_options):
    return dict(
        units=units, activation=activation, layer_norm=layer_norm, **initializer_options
    )


@pytest.fixture
def fc_kwargs(in_features, kwargs):
    return dict(in_features=in_features, **kwargs)


def test_fully_connected(fc_kwargs, torch_script):
    module = FullyConnected(**fc_kwargs)
    if torch_script:
        module = torch.jit.script(module)

    inputs = torch.randn(1, fc_kwargs["in_features"])
    module(inputs)


@pytest.fixture(params=((1, 1), (2, 2), (4, 4)))
def obs_action_dim(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("DelayAction", "ConcatAction"))
def delay_action(request):
    return request.param


@pytest.fixture
def sae_kwargs(obs_action_dim, delay_action, kwargs):
    obs_dim, action_dim = obs_action_dim
    return dict(
        obs_dim=obs_dim, action_dim=action_dim, delay_action=delay_action, **kwargs
    )


def test_state_action_encoder(sae_kwargs, torch_script):
    module = StateActionEncoder(**sae_kwargs)
    if torch_script:
        module = torch.jit.script(module)

    obs, action = (
        torch.randn(1, sae_kwargs["obs_dim"]),
        torch.randn(1, sae_kwargs["action_dim"]),
    )
    module(obs, action)
