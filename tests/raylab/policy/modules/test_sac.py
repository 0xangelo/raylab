import pytest
import torch
import torch.nn as nn

import raylab.torch.nn as nnx
from raylab.policy.modules.actor.policy.stochastic import MLPContinuousPolicy
from raylab.policy.modules.sac import SAC


@pytest.fixture
def spec_cls():
    return SAC.spec_cls


@pytest.fixture
def module(obs_space, action_space, spec_cls):
    return SAC(obs_space, action_space, spec_cls())


def test_spec(spec_cls):
    default_config = spec_cls().to_dict()

    for key in ["actor", "critic", "initializer"]:
        assert key in default_config


def test_init(module):
    assert isinstance(module, nn.Module)

    for attr in "actor alpha critics target_critics".split():
        assert hasattr(module, attr)


def test_actor_network(module):
    assert isinstance(module.actor, MLPContinuousPolicy)
    assert isinstance(module.actor.params[-1], nnx.PolicyNormalParams)


def test_script(module):
    torch.jit.script(module)
