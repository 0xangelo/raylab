import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch


@pytest.fixture(params=(True, False))
def clipped_double_q(request):
    return request.param


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, clipped_double_q):
    config = {"clipped_double_q": clipped_double_q, "polyak": 0.5}
    return policy_and_batch_fn(config)


def test_target_value_output(policy_and_batch):
    policy, batch = policy_and_batch
    next_vals = policy.module.target_critic(
        batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
    )
    assert next_vals.shape == (10, 1)
    assert next_vals.dtype == torch.float32
    if policy.config["clipped_double_q"]:
        next_vals = policy.module.target_twin_critic(
            batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS]
        )
        assert next_vals.shape == (10, 1)
        assert next_vals.dtype == torch.float32

    targets = policy._compute_critic_targets(batch, policy.module, policy.config)
    assert targets.shape == (10,)
    assert targets.dtype == torch.float32
    assert torch.allclose(
        targets[batch[SampleBatch.DONES]],
        batch[SampleBatch.REWARDS][batch[SampleBatch.DONES]],
    )

    policy.module.zero_grad()
    targets.mean().backward()
    target_params = set(policy.module.target_critic.parameters())
    if policy.config["clipped_double_q"]:
        target_params.update(set(policy.module.target_twin_critic.parameters()))
    target_params.update(set(policy.module.policy.parameters()))
    target_params.update({policy.module.log_alpha})
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in set(policy.module.parameters()) - target_params)


def test_critic_loss(policy_and_batch):
    policy, batch = policy_and_batch
    loss, _ = policy.compute_critic_loss(batch, policy.module, policy.config)

    assert loss.shape == ()
    assert loss.dtype == torch.float32

    params = set(policy.module.critic.parameters())
    if policy.config["clipped_double_q"]:
        params.update(set(policy.module.twin_critic.parameters()))
    loss.backward()
    assert all(p.grad is not None for p in params)
    assert all(p.grad is None for p in policy.module.parameters() if p not in params)


def test_target_params_update(policy_and_batch):
    policy, _ = policy_and_batch
    params = list(policy.module.critic.parameters())
    target_params = list(policy.module.target_critic.parameters())
    if policy.config["clipped_double_q"]:
        params += list(policy.module.twin_critic.parameters())
        target_params += list(policy.module.target_twin_critic.parameters())
    assert all(torch.allclose(p, q) for p, q in zip(params, target_params))

    old_params = [p.clone() for p in target_params]
    for param in params:
        param.data.add_(torch.ones_like(param))
    policy.update_targets()
    assert all(not torch.allclose(p, q) for p, q in zip(target_params, old_params))
