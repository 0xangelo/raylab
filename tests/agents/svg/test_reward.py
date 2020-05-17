# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, svg_policy):
    return policy_and_batch_fn(svg_policy, {})


def test_reward_reproduce(policy_and_batch):
    policy, batch = policy_and_batch

    rews = batch[SampleBatch.REWARDS]
    _rews = policy.reward(
        batch[SampleBatch.CUR_OBS],
        batch[SampleBatch.ACTIONS],
        batch[SampleBatch.NEXT_OBS],
    )
    assert _rews.shape == rews.shape
    assert _rews.dtype == rews.dtype
    assert torch.allclose(_rews, rews, atol=1e-6)
