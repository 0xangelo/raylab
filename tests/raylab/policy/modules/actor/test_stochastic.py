# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch


@pytest.fixture(scope="module")
def module_cls():
    from raylab.policy.modules.actor.stochastic import StochasticActor

    return StochasticActor


@pytest.fixture(params=(True, False), ids=lambda x: f"InputDependentScale({x})")
def input_dependent_scale(request):
    return request.param


@pytest.fixture
def cont_spec(module_cls, input_dependent_scale):
    return module_cls.spec_cls(input_dependent_scale=input_dependent_scale)


@pytest.fixture
def spec(module_cls):
    return module_cls.spec_cls()


@pytest.fixture
def disc_module(module_cls, obs_space, disc_space, spec, torch_script):
    mod = module_cls(obs_space, disc_space, spec)
    return torch.jit.script(mod) if torch_script else mod


@pytest.fixture
def cont_module(module_cls, obs_space, cont_space, spec, torch_script):
    mod = module_cls(obs_space, cont_space, spec)
    return torch.jit.script(mod) if torch_script else mod


@pytest.fixture
def module(module_cls, obs_space, action_space, spec, torch_script):
    mod = module_cls(obs_space, action_space, spec)
    return torch.jit.script(mod) if torch_script else mod


def test_init(module):
    for attr in "policy alpha".split():
        assert hasattr(module, attr)
