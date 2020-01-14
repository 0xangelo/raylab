# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.algorithms.registry import ALGORITHMS as ALGS


@pytest.fixture
def mapo_trainer():
    return ALGS["MAPO"]()


@pytest.fixture
def mapo_policy(mapo_trainer):
    return mapo_trainer._policy


@pytest.fixture
def reward_fn():
    def rew_fn(_, actions, next_obs):
        reward_dist = -torch.norm(next_obs, dim=-1)
        reward_ctrl = -torch.sum(actions ** 2, dim=-1)
        return reward_dist + reward_ctrl

    return rew_fn


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_, mapo_policy, reward_fn):
    def make_policy_and_batch(config):
        policy, batch = policy_and_batch_(mapo_policy, config)
        policy.set_reward_fn(reward_fn)
        batch[SampleBatch.REWARDS] = reward_fn(
            batch[SampleBatch.CUR_OBS],
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.NEXT_OBS],
        )
        return policy, batch

    return make_policy_and_batch
