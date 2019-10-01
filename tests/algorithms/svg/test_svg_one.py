import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.utils.debug import fake_batch


@pytest.fixture
def reward_fn():
    return lambda s, a, s_: a.sum(dim=-1) * s_.mean(dim=-1)


@pytest.fixture
def policy_and_batch(svg_one_policy, obs_space, action_space, reward_fn):
    policy = svg_one_policy(obs_space, action_space, {})
    policy.set_reward_fn(reward_fn)
    batch = policy._lazy_tensor_dict(fake_batch(obs_space, action_space, batch_size=10))
    batch[SampleBatch.REWARDS] = reward_fn(
        batch[SampleBatch.CUR_OBS],
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
    )
    return policy, batch


def test_truncated_svg(policy_and_batch):
    policy, batch = policy_and_batch

    td_targets = policy._compute_policy_td_targets(batch)
    assert td_targets.shape == (10,)
    assert td_targets.dtype == torch.float32
    assert torch.allclose(
        td_targets[batch[SampleBatch.DONES]],
        batch[SampleBatch.REWARDS][batch[SampleBatch.DONES]],
    )

    td_targets.mean().backward()
    assert all(p.grad is not None for p in policy.module.policy.parameters())
