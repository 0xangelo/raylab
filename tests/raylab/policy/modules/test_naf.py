import pytest
import torch
from torch import nn

from raylab.policy.modules.naf import NAF


@pytest.fixture
def spec_cls():
    return NAF.spec_cls


@pytest.fixture(params=(True, False), ids=lambda x: f"DoubleQ({x})")
def double_q(request):
    return request.param


@pytest.fixture
def spec(spec_cls, double_q):
    return spec_cls(double_q=double_q)


@pytest.fixture
def module(obs_space, action_space, spec):
    return NAF(obs_space, action_space, spec)


def test_spec(spec_cls):
    default_config = spec_cls().to_dict()

    for key in "policy".split():
        assert key in default_config


def test_init(module):
    assert isinstance(module, nn.Module)

    for attr in "actor behavior critics vcritics target_vcritics".split():
        assert hasattr(module, attr)


def test_script(module):
    torch.jit.script(module)
