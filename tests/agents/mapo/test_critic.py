# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import torch.nn as nn
from ray.rllib import SampleBatch


@pytest.fixture(params=(True, False), ids=("DoubleQ", "SingleQ"))
def clipped_double_q(request):
    return request.param


@pytest.fixture(params=(True, False), ids=("SmoothTarget", "HardTarget"))
def smooth_target_policy(request):
    return request.param


@pytest.fixture
def config(clipped_double_q, smooth_target_policy):
    return {
        "clipped_double_q": clipped_double_q,
        "module": {"actor": {"smooth_target_policy": smooth_target_policy}},
    }


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, config):
    policy, batch = policy_and_batch_fn(config)
    for par in policy.module.parameters():
        par.grad = None
    return policy, batch


def test_critic_targets(policy_and_batch):
    policy, batch = policy_and_batch

    targets = policy.critic_targets(batch)
    assert targets.shape == (10,)
    assert targets.dtype == torch.float32
    assert torch.allclose(
        targets[batch[SampleBatch.DONES]],
        batch[SampleBatch.REWARDS][batch[SampleBatch.DONES]],
    )

    policy.module.zero_grad()
    targets.mean().backward()
    target_params = set(policy.module.target_critics.parameters())
    target_params.update(set(policy.module.target_actor.parameters()))
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in set(policy.module.parameters()) - target_params)


def test_critic_loss(policy_and_batch):
    policy, batch = policy_and_batch
    loss, info = policy.critic_loss(batch)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert isinstance(info, dict)

    params = set(policy.module.critics.parameters())
    loss.backward()
    assert all(p.grad is not None for p in params)
    assert all(p.grad is None for p in set(policy.module.parameters()) - params)

    vals = [
        m(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
        for m in policy.module.critics
    ]
    concat_vals = torch.cat(vals, dim=-1)
    targets = torch.randn_like(vals[0])
    loss_fn = nn.MSELoss()
    assert torch.allclose(
        loss_fn(concat_vals, targets.expand_as(concat_vals)),
        sum(loss_fn(val, targets) for val in vals) / len(vals),
    )


def test_target_params_update(policy_and_batch):
    policy, _ = policy_and_batch
    params = list(policy.module.critics.parameters())
    target_params = list(policy.module.target_critics.parameters())
    assert all(torch.allclose(p, q) for p, q in zip(params, target_params))

    old_params = [p.clone() for p in target_params]
    for param in params:
        param.data.add_(torch.ones_like(param))
    policy.update_targets("critics", "target_critics")
    assert all(not torch.allclose(p, q) for p, q in zip(target_params, old_params))
