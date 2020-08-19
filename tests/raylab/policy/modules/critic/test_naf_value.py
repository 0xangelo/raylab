import pytest
import torch

from raylab.policy.modules.actor.policy.deterministic import MLPDeterministicPolicy
from raylab.policy.modules.critic.naf_value import NAFQValue


@pytest.fixture
def policy_cls():
    return MLPDeterministicPolicy


@pytest.fixture
def policy(policy_cls, obs_space, action_space):
    return policy_cls(obs_space, action_space, policy_cls.spec_cls())


@pytest.fixture
def q_value(action_space, policy):
    return NAFQValue(action_space, policy)


def test_forward(q_value, obs, action):
    value = q_value(obs, action)
    assert torch.is_tensor(value)
    assert value.shape == (len(obs),)

    value.sum().backward()
    assert any([p.grad is not None for p in q_value.parameters()])
