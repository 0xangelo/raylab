import pytest
import torch


@pytest.fixture(scope="module")
def module_cls():
    from raylab.torch.nn.actor.stochastic import StochasticActor

    return StochasticActor


@pytest.fixture
def spec(module_cls):
    return module_cls.spec_cls()


@pytest.fixture
def module(module_cls, obs_space, action_space, spec, torch_script):
    mod = module_cls(obs_space, action_space, spec)
    return torch.jit.script(mod) if torch_script else mod


def test_init(module):
    for attr in "policy alpha".split():
        assert hasattr(module, attr)
