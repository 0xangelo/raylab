import pytest
import torch
from ray.rllib import SampleBatch

from raylab.policy.losses.svg import OneStepSVG
from raylab.policy.losses.svg import ReproduceRewards


@pytest.fixture
def actor(stochastic_policy):
    return stochastic_policy


@pytest.fixture
def critic(state_critics):
    main, _ = state_critics
    return main


@pytest.fixture
def loss_fn(model, actor, critic, reward_fn):
    loss = OneStepSVG(model, actor, critic)
    loss.set_reward_fn(reward_fn)
    return loss


def test_init(loss_fn: OneStepSVG):
    assert loss_fn.batch_keys == (
        SampleBatch.CUR_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.NEXT_OBS,
        SampleBatch.DONES,
        loss_fn.IS_RATIOS,
    )


def test_truncated_svg(loss_fn, is_batch, actor, reward_fn):
    batch = is_batch(loss_fn.IS_RATIOS)

    obs, action, new_obs, done, _ = loss_fn.unpack_batch(batch)
    rew = reward_fn(obs, action, new_obs)
    state_vals = loss_fn.one_step_reproduced_state_value(obs, action, new_obs, done)
    assert state_vals.shape == rew.shape
    assert state_vals.dtype == rew.dtype
    assert torch.allclose(state_vals[done], rew[done], atol=1e-6)

    state_vals.mean().backward()
    assert all(p.grad is not None for p in actor.parameters())


@pytest.fixture
def repr_rewards(actor, model, reward_fn):
    return ReproduceRewards(policy=actor, model=model, reward_fn=reward_fn)


@pytest.fixture
def consistent_batch(batch):
    batch[SampleBatch.CUR_OBS][1:] = batch[SampleBatch.NEXT_OBS][:-1]
    return batch


@torch.no_grad()
def test_reproduce_rewards(repr_rewards, consistent_batch, reward_fn):
    batch = consistent_batch
    assert repr_rewards.reward_fn is reward_fn

    obs, act, rew = repr_rewards(
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
        batch[SampleBatch.CUR_OBS][0],
    )

    def allclose(a, b):  # pylint:disable=invalid-name
        return torch.allclose(a, b, atol=1e-6)

    assert allclose(obs, batch[SampleBatch.NEXT_OBS])
    assert allclose(
        obs,
        torch.cat([batch[SampleBatch.CUR_OBS][1:], batch[SampleBatch.NEXT_OBS][-1:]]),
    )
    assert allclose(act, batch[SampleBatch.ACTIONS])
    target = reward_fn(
        batch[SampleBatch.CUR_OBS],
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
    )
    target_ = reward_fn(batch[SampleBatch.CUR_OBS], act, obs)
    assert allclose(target, target_)
    assert allclose(rew, target)


def test_propagates_gradients(repr_rewards, consistent_batch, actor):
    batch = consistent_batch
    _, _, rew = repr_rewards(
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
        batch[SampleBatch.CUR_OBS][0],
    )

    rew.sum().backward()

    assert all(p.grad is not None for p in actor.parameters())
