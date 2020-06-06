# pylint:disable=missing-docstring,redefined-outer-name,protected-access
import math

import pytest
import torch

from raylab.envs import get_reward_fn
from raylab.envs import get_termination_fn
from raylab.policy import EnvFnMixin
from raylab.policy import TorchPolicy
from raylab.utils.debug import fake_space_samples


class DummyPolicy(EnvFnMixin, TorchPolicy):
    # pylint:disable=all
    @staticmethod
    def get_default_config():
        return {"module": {"type": "OnPolicyActorCritic"}}

    def make_optimizer(self):
        pass


@pytest.fixture
def reward_fn():
    def func(obs, act, new_obs):
        return act.norm(p=1, dim=-1)

    return func


@pytest.fixture
def termination_fn():
    def func(obs, act, new_obs):
        return torch.randn(obs.shape[:-1]) > 0

    return func


@pytest.fixture
def dynamics_fn():
    def func(obs, act):
        sample = torch.randn_like(obs)
        log_prob = torch.sum(
            -(sample ** 2) / 2
            - torch.ones_like(obs).log()
            - math.log(math.sqrt(2 * math.pi)),
            dim=-1,
        )
        return sample, log_prob

    return func


@pytest.fixture
def policy(obs_space, action_space):
    return DummyPolicy(obs_space, action_space, {})


def test_init(policy):
    assert hasattr(policy, "reward_fn")
    assert hasattr(policy, "termination_fn")
    assert hasattr(policy, "dynamics_fn")


def test_set_reward_from_config(policy, envs):  # pylint:disable=unused-argument
    obs_space, action_space = policy.observation_space, policy.action_space
    batch_size = 10
    obs = fake_space_samples(obs_space, batch_size=batch_size)
    act = fake_space_samples(action_space, batch_size=batch_size)
    new_obs = fake_space_samples(obs_space, batch_size=batch_size)
    obs, act, new_obs = map(policy.convert_to_tensor, (obs, act, new_obs))

    policy.set_reward_from_config("MockEnv", {})

    original_fn = get_reward_fn("MockEnv", {})
    expected_rew = original_fn(obs, act, new_obs)
    rew = policy.reward_fn(obs, act, new_obs)

    assert torch.allclose(rew, expected_rew)


def test_set_termination_from_config(policy, envs):  # pylint:disable=unused-argument
    obs_space, action_space = policy.observation_space, policy.action_space
    batch_size = 10
    obs = fake_space_samples(obs_space, batch_size=batch_size)
    act = fake_space_samples(action_space, batch_size=batch_size)
    new_obs = fake_space_samples(obs_space, batch_size=batch_size)
    obs, act, new_obs = map(policy.convert_to_tensor, (obs, act, new_obs))

    policy.set_termination_from_config("MockEnv", {})

    done = policy.termination_fn(obs, act, new_obs)
    assert torch.is_tensor(done)
    assert done.dtype == torch.bool
    assert done.shape == obs.shape[:-1]


def test_set_reward_from_callable(policy, reward_fn):
    policy.set_reward_from_callable(reward_fn)
    assert hasattr(policy, "reward_fn")
    assert policy.reward_fn is reward_fn


def test_set_termination_from_callable(policy, termination_fn):
    policy.set_termination_from_callable(termination_fn)
    assert hasattr(policy, "termination_fn")
    assert policy.termination_fn is termination_fn


def test_set_dynamics_from_callable(policy, dynamics_fn):
    policy.set_dynamics_from_callable(dynamics_fn)
    assert hasattr(policy, "dynamics_fn")
    assert policy.dynamics_fn is dynamics_fn
