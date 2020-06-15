# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn
from ray.rllib import SampleBatch

import raylab.utils.dictionaries as dutil
from raylab.losses import ClippedDoubleQLearning
from raylab.modules.mixins.action_value_mixin import ActionValueFunction
from raylab.modules.mixins.deterministic_actor_mixin import DeterministicPolicy
from raylab.utils.debug import fake_batch


@pytest.fixture(params=(1, 2), ids=(f"Critics({n})" for n in (1, 2)))
def critics(request, obs_space, action_space):
    def critic():
        return ActionValueFunction.from_scratch(
            obs_space.shape[0], action_space.shape[0], units=(32,)
        )

    critics = nn.ModuleList([critic() for _ in range(request.param)])
    target_critics = nn.ModuleList([critic() for _ in range(request.param)])
    target_critics.load_state_dict(critics.state_dict())
    return critics, target_critics


@pytest.fixture
def target_policy(obs_space, action_space):
    policy = DeterministicPolicy.from_scratch(
        obs_space, action_space, {"beta": 1.2, "encoder": {"units": (32,)}}
    )
    return policy


@pytest.fixture
def loss_fn(critics, target_policy):
    critics, target_critics = critics
    return ClippedDoubleQLearning(critics, target_critics, target_policy, gamma=0.99)


@pytest.fixture(scope="module")
def batch(obs_space, action_space):
    samples = fake_batch(obs_space, action_space, batch_size=256)
    return {k: torch.from_numpy(v) for k, v in samples.items()}


def test_init(loss_fn):
    assert hasattr(loss_fn, "batch_keys")
    assert hasattr(loss_fn, "gamma")


def test_target_value(loss_fn, batch, critics, target_policy):
    critics, target_critics = critics
    modules = nn.ModuleList([critics, target_critics, target_policy])

    rewards, next_obs, dones = dutil.get_keys(
        batch, SampleBatch.REWARDS, SampleBatch.NEXT_OBS, SampleBatch.DONES
    )
    targets = loss_fn.critic_targets(rewards, next_obs, dones)
    assert targets.shape == (len(next_obs),)
    assert targets.dtype == torch.float32
    assert torch.allclose(targets[dones], rewards[dones])

    modules.zero_grad()
    targets.mean().backward()
    target_params = set(target_critics.parameters())
    target_params.update(set(target_policy.parameters()))
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in set(critics.parameters()))


def test_critic_loss(loss_fn, batch, critics, target_policy):
    critics, target_critics = critics

    loss, info = loss_fn(batch)
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert isinstance(info, dict)

    loss.backward()
    assert all(p.grad is not None for p in set(critics.parameters()))
    assert all(
        p.grad is None
        for p in set.union(
            set(target_critics.parameters()), set(target_policy.parameters())
        )
    )

    obs, acts = dutil.get_keys(batch, SampleBatch.CUR_OBS, SampleBatch.ACTIONS)
    vals = [m(obs, acts) for m in critics]
    concat_vals = torch.cat(vals, dim=-1)
    targets = torch.randn_like(vals[0])
    loss_fn = nn.MSELoss()
    assert torch.allclose(
        loss_fn(concat_vals, targets.expand_as(concat_vals)),
        sum(loss_fn(val, targets) for val in vals) / len(vals),
    )
