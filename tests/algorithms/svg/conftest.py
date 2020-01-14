# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
from ray.rllib.policy.sample_batch import SampleBatch

from raylab.algorithms.registry import ALGORITHMS as ALGS


@pytest.fixture(params=[ALGS[k] for k in "SVG(1) SVG(inf)".split()])
def svg_trainer(request):
    return request.param()


@pytest.fixture
def svg_policy(svg_trainer):
    return svg_trainer._policy


@pytest.fixture
def svg_one_trainer():
    return ALGS["SVG(1)"]()


@pytest.fixture
def svg_one_policy(svg_one_trainer):
    return svg_one_trainer._policy


@pytest.fixture
def svg_inf_trainer():
    return ALGS["SVG(inf)"]()


@pytest.fixture
def svg_inf_policy(svg_inf_trainer):
    return svg_inf_trainer._policy


@pytest.fixture
def reward_fn():
    def rew_fn(_, actions, next_obs):
        reward_dist = -torch.norm(next_obs, dim=-1)
        reward_ctrl = -torch.sum(actions ** 2, dim=-1)
        return reward_dist + reward_ctrl

    return rew_fn


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_, reward_fn):
    def make_policy_and_batch(policy_cls, config):
        policy, batch = policy_and_batch_(policy_cls, config)
        policy.set_reward_fn(reward_fn)
        batch[SampleBatch.REWARDS] = reward_fn(
            batch[SampleBatch.CUR_OBS],
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.NEXT_OBS],
        )
        return policy, batch

    return make_policy_and_batch
