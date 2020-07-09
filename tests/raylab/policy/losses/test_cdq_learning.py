import pytest
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

import raylab.utils.dictionaries as dutil
from raylab.policy.losses import ClippedDoubleQLearning
from raylab.policy.losses import SoftCDQLearning


@pytest.fixture
def critics(action_critics):
    return action_critics[0]


@pytest.fixture
def target_critics(action_critics):
    return action_critics[1]


@pytest.fixture
def target_policy(deterministic_policies):
    _, target_policy = deterministic_policies
    return target_policy


@pytest.fixture
def cdq_loss(critics, target_critics, target_policy):
    return ClippedDoubleQLearning(critics, target_critics, target_policy)


def test_init(cdq_loss):
    assert hasattr(cdq_loss, "batch_keys")
    assert hasattr(cdq_loss, "gamma")


def test_target_value(cdq_loss, batch, critics, target_critics, target_policy):
    modules = nn.ModuleList([critics, target_critics, target_policy])

    rewards, next_obs, dones = dutil.get_keys(
        batch, SampleBatch.REWARDS, SampleBatch.NEXT_OBS, SampleBatch.DONES
    )
    targets = cdq_loss.critic_targets(rewards, next_obs, dones)
    assert targets.shape == (len(next_obs),)
    assert targets.dtype == torch.float32
    assert torch.allclose(targets[dones], rewards[dones])

    modules.zero_grad()
    targets.mean().backward()
    target_params = set(target_critics.parameters())
    target_params.update(set(target_policy.parameters()))
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in set(critics.parameters()))


def test_critic_loss(cdq_loss, batch, critics, target_critics, target_policy):
    loss, info = cdq_loss(batch)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert isinstance(info, dict)

    loss.backward()
    aux_params = set.union(
        set(target_critics.parameters()), set(target_policy.parameters())
    )
    assert all(p.grad is not None for p in set(critics.parameters()))
    assert all(p.grad is None for p in aux_params)

    obs, acts = dutil.get_keys(batch, SampleBatch.CUR_OBS, SampleBatch.ACTIONS)
    vals = [m(obs, acts) for m in critics]
    concat_vals = torch.cat(vals, dim=-1)
    targets = torch.randn_like(vals[0])
    cdq_loss = nn.MSELoss()
    assert torch.allclose(
        cdq_loss(concat_vals, targets.expand_as(concat_vals)),
        sum(cdq_loss(val, targets) for val in vals) / len(vals),
    )


@pytest.fixture
def actor(stochastic_policy):
    return stochastic_policy


@pytest.fixture
def soft_cdq_loss(critics, target_critics, actor):
    return SoftCDQLearning(critics, target_critics, actor.sample)


def test_soft_critic_targets(soft_cdq_loss, batch, critics, target_critics, actor):
    loss_fn = soft_cdq_loss

    rewards, next_obs, dones = dutil.get_keys(
        batch, SampleBatch.REWARDS, SampleBatch.NEXT_OBS, SampleBatch.DONES
    )
    targets = loss_fn.critic_targets(rewards, next_obs, dones)
    assert targets.shape == (len(next_obs),)
    assert targets.dtype == torch.float32
    assert torch.allclose(targets[dones], batch[SampleBatch.REWARDS][dones])

    targets.mean().backward()
    target_params = set(target_critics.parameters())
    target_params.update(set(actor.parameters()))
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in critics.parameters())


def test_soft_critic_loss(soft_cdq_loss, batch, critics, target_critics, actor):
    loss_fn = soft_cdq_loss

    loss, info = loss_fn(batch)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert isinstance(info, dict)

    params = set(critics.parameters())
    aux_params = set.union(set(target_critics.parameters()), set(actor.parameters()))
    loss.backward()
    assert all(p.grad is not None for p in params)
    assert all(p.grad is None for p in aux_params)

    obs, acts = dutil.get_keys(batch, SampleBatch.CUR_OBS, SampleBatch.ACTIONS)
    vals = [m(obs, acts) for m in critics]
    concat_vals = torch.cat(vals, dim=-1)
    targets = torch.randn_like(vals[0])
    loss_fn = nn.MSELoss()
    assert torch.allclose(
        loss_fn(concat_vals, targets.expand_as(concat_vals)),
        sum(loss_fn(val, targets) for val in vals) / len(vals),
    )
