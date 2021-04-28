import pytest
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

import raylab.utils.dictionaries as dutil
from raylab.policy.losses import FittedQLearning
from raylab.policy.modules.critic import HardValue, SoftValue


@pytest.fixture
def critics(action_critics):
    return action_critics[0]


@pytest.fixture
def target_critic(deterministic_policies, action_critics):
    _, target_policy = deterministic_policies
    _, target_critics = action_critics
    module = HardValue(target_policy, target_critics)
    module.requires_grad_(True)
    return module


@pytest.fixture
def cdq_loss(critics, target_critic):
    return FittedQLearning(critics, target_critic)


def test_init(cdq_loss):
    assert hasattr(cdq_loss, "batch_keys")
    assert hasattr(cdq_loss, "gamma")


def test_target_value(cdq_loss, batch, critics, target_critic):
    modules = nn.ModuleList([critics, target_critic])

    rewards, next_obs, dones = dutil.get_keys(
        batch, SampleBatch.REWARDS, SampleBatch.NEXT_OBS, SampleBatch.DONES
    )
    targets = cdq_loss.critic_targets(rewards, next_obs, dones)
    assert torch.is_tensor(targets)
    assert targets.shape == (len(next_obs),)
    assert targets.dtype == torch.float32
    assert torch.allclose(targets[dones], rewards[dones])

    modules.zero_grad()
    targets.mean().backward()
    target_params = set(target_critic.parameters())
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in set(critics.parameters()))


def test_critic_loss(cdq_loss, batch, critics, target_critic):
    loss, info = cdq_loss(batch)
    assert torch.is_tensor(loss)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert isinstance(info, dict)

    loss.backward()
    aux_params = set(target_critic.parameters())
    assert all([any([p.grad is not None for p in c.parameters()]) for c in critics])
    assert all(p.grad is None for p in aux_params)


@pytest.fixture
def soft_target(stochastic_policy, action_critics, alpha_module):
    _, target_critics = action_critics
    return SoftValue(stochastic_policy, target_critics, alpha=alpha_module)


@pytest.fixture
def soft_cdq_loss(critics, soft_target):
    return FittedQLearning(critics, soft_target)


def test_soft_critic_targets(soft_cdq_loss, rew, done, new_obs, critics, soft_target):
    # pylint:disable=too-many-arguments
    loss_fn = soft_cdq_loss

    targets = loss_fn.critic_targets(rew, new_obs, done)
    assert torch.is_tensor(targets)
    assert targets.shape == (len(new_obs),)
    assert targets.dtype == torch.float32
    assert torch.allclose(targets[done], rew[done])

    targets.mean().backward()
    assert any([p.grad is not None for p in soft_target.parameters()])
    assert all([p.grad is None for p in critics.parameters()])


def test_soft_critic_loss(soft_cdq_loss, batch, critics, soft_target):
    loss_fn = soft_cdq_loss

    loss, info = loss_fn(batch)
    assert torch.is_tensor(loss)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert isinstance(info, dict)

    params = [set(c.parameters()) for c in critics]
    aux_params = set(soft_target.parameters())
    loss.backward()
    assert all([any([p.grad is not None for p in pars]) for pars in params])
    assert all([p.grad is None for p in aux_params])
