# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, svg_one_policy):
    return policy_and_batch_fn(svg_one_policy, {})


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
    assert all(p.grad is not None for p in policy.module.actor.parameters())
