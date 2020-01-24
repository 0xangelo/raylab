# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
import torch
import numpy as np


@pytest.fixture(params=((), (1,), (2,)))
def sample_shape(request):
    return request.param


@pytest.fixture(params=((), (1,), (2,)))
def batch_shape(request):
    return request.param


@pytest.fixture
def markovian_ib(envs):
    return envs["IndustrialBenchmark"]({"observation": "markovian"})


@pytest.fixture
def classic_reward_ib(envs):
    return envs["IndustrialBenchmark"]({"reward_type": "classic"})


@pytest.fixture
def delta_reward_ib(envs):
    return envs["IndustrialBenchmark"]({"reward_type": "delta"})


def test_reward_type(classic_reward_ib, delta_reward_ib):
    classic_reward_ib.seed(42)
    classic_reward_ib.reset()
    delta_reward_ib.seed(42)
    delta_reward_ib.reset()

    _, rew, _, _ = classic_reward_ib.step([0.5] * 3)
    _, rew2, _, _ = classic_reward_ib.step([0.5] * 3)
    _, _, _, _ = delta_reward_ib.step([0.5] * 3)
    _, rew_, _, _ = delta_reward_ib.step([0.5] * 3)

    assert np.allclose(rew2 - rew, rew_)


def test_transition_fn(markovian_ib, batch_shape, sample_shape):
    env = markovian_ib
    obs = torch.as_tensor(env.reset())
    obs = obs.expand(batch_shape + obs.shape)
    act = torch.as_tensor(env.action_space.sample())
    act = act.expand(batch_shape + act.shape)
    act.requires_grad_()

    next_obs, logp = env.transition_fn(obs, act, sample_shape=sample_shape)
    assert logp is None
    assert next_obs.shape == sample_shape + obs.shape
    # hack to get single observation regardless of batch or sample size
    single_obs = next_obs[tuple(0 for _ in next_obs.shape[:-1])]
    assert single_obs.detach().numpy() in env.observation_space

    assert next_obs.grad_fn is not None
