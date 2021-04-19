import pytest
import torch

from raylab.policy.losses import DynaQLearning
from raylab.policy.modules.actor import Alpha
from raylab.policy.modules.critic import ClippedQValue, SoftValue


@pytest.fixture
def critics(action_critics):
    return action_critics[0]


@pytest.fixture
def actor(stochastic_policy):
    return stochastic_policy


@pytest.fixture
def target_critic(actor, action_critics):
    return SoftValue(actor, ClippedQValue(action_critics[1]), Alpha(1.0))


@pytest.fixture
def model_samples():
    return 2


@pytest.fixture
def dyna_loss(critics, actor, models, target_critic):
    return DynaQLearning(critics, actor, models, target_critic)


def test_dyna_cdq(dyna_loss, reward_fn, termination_fn, batch, critics):
    loss_fn = dyna_loss
    loss_fn.set_reward_fn(reward_fn)
    loss_fn.set_termination_fn(termination_fn)

    loss, info = loss_fn(batch)
    assert torch.is_tensor(loss)
    assert loss.shape == ()
    assert isinstance(info, dict)
    assert "loss(critics)" in info

    loss.backward()
    assert all([p.grad is not None for p in critics.parameters()])
