import pytest
import torch
from ray.rllib import SampleBatch

from raylab.policy.losses import ISFittedVIteration


@pytest.fixture
def critic(state_critics):
    main, _ = state_critics
    return main


@pytest.fixture
def target_critic(state_critics):
    _, target = state_critics
    return target


@pytest.fixture
def loss_fn(critic, target_critic):
    return ISFittedVIteration(critic, target_critic)


def test_init(loss_fn: ISFittedVIteration):
    assert loss_fn.batch_keys == (
        SampleBatch.CUR_OBS,
        loss_fn.IS_RATIOS,
        SampleBatch.NEXT_OBS,
        SampleBatch.REWARDS,
        SampleBatch.DONES,
    )


def test_compute_value_targets(
    loss_fn: ISFittedVIteration, is_batch, critic, target_critic
):
    batch = is_batch(loss_fn.IS_RATIOS)

    obs, _, new_obs, rew, done = loss_fn.unpack_batch(batch)
    targets = loss_fn.sampled_one_step_state_values(obs, new_obs, rew, done)
    assert targets.shape == rew.shape
    assert targets.dtype == rew.dtype
    assert torch.allclose(targets[done], rew[done])

    critic.zero_grad()
    target_critic.zero_grad()
    targets.mean().backward()
    target_params = set(target_critic.parameters())
    other_params = set(critic.parameters())
    assert all(p.grad is not None for p in target_params)
    assert all(p.grad is None for p in other_params)


def test_importance_sampling_weighted_loss(
    loss_fn: ISFittedVIteration, is_batch, critic, target_critic
):
    batch = is_batch(loss_fn.IS_RATIOS)

    loss, info = loss_fn(batch)
    loss.backward()
    value_params = set(critic.parameters())
    other_params = set(target_critic.parameters())
    assert all(p.grad is not None for p in value_params)
    assert all(p.grad is None for p in other_params)

    assert "loss(critic)" in info
