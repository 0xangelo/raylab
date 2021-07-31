import pytest
import torch

from raylab.policy.modules.svg import SVG
from raylab.torch.nn.actor import StochasticPolicy
from raylab.torch.nn.critic import VValue
from raylab.torch.nn.model import StochasticModel


@pytest.fixture
def spec_cls():
    return SVG.spec_cls


@pytest.fixture
def module(obs_space, action_space, spec_cls):
    return SVG(obs_space, action_space, spec_cls())


def test_attrs(module):
    for attr in "model actor critic target_critic".split():
        assert hasattr(module, attr)

    assert isinstance(module.model, StochasticModel)
    assert isinstance(module.actor, StochasticPolicy)
    assert isinstance(module.critic, VValue)
    assert isinstance(module.target_critic, VValue)


def test_model_params(module, obs, action, next_obs):
    params = module.model(obs, action)
    assert "loc" in params
    assert "scale" in params

    loc, scale = params["loc"], params["scale"]
    assert loc.shape == next_obs.shape
    assert scale.shape == next_obs.shape
    assert loc.dtype == torch.float32
    assert scale.dtype == torch.float32

    parameters = set(module.model.parameters())
    for par in parameters:
        par.grad = None
    loc.mean().backward()
    assert any([p.grad is not None for p in parameters])
    assert all([p.grad is None for p in set(module.parameters()) - parameters])

    for par in parameters:
        par.grad = None
    module.model(obs, action)["scale"].mean().backward()
    assert any([p.grad is not None for p in parameters])
    assert all([p.grad is None for p in set(module.parameters()) - parameters])
