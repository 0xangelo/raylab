import pytest

from raylab.policy.modules.actor import StochasticPolicy
from raylab.policy.modules.critic import VValue
from raylab.policy.modules.trpo import TRPO


@pytest.fixture
def spec_cls():
    return TRPO.spec_cls


@pytest.fixture
def module(obs_space, action_space, spec_cls):
    return TRPO(obs_space, action_space, spec_cls())


def test_attrs(module):
    for attr in "actor critic".split():
        assert hasattr(module, attr)

    assert isinstance(module.actor, StochasticPolicy)
    assert isinstance(module.critic, VValue)
