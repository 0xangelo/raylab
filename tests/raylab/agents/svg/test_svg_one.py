# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib import SampleBatch

from raylab.utils.dictionaries import get_keys


@pytest.fixture
def policy_and_batch(policy_and_batch_fn, svg_one_policy):
    policy, batch = policy_and_batch_fn(svg_one_policy, {})
    policy.set_reward_from_config(policy.config["env"], policy.config["env_config"])
    return policy, batch


def test_truncated_svg(policy_and_batch):
    policy, batch = policy_and_batch

    obs, actions, next_obs, rewards, dones = get_keys(
        batch,
        SampleBatch.CUR_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.NEXT_OBS,
        SampleBatch.REWARDS,
        SampleBatch.DONES,
    )
    state_vals = policy.loss_actor.one_step_reproduced_state_value(
        obs, actions, next_obs, dones
    )
    assert state_vals.shape == (10,)
    assert state_vals.dtype == torch.float32
    assert torch.allclose(state_vals[dones], rewards[dones],)

    state_vals.mean().backward()
    assert all(p.grad is not None for p in policy.module.actor.parameters())
