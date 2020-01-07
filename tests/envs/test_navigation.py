# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import numpy as np
import torch

from raylab.utils.pytorch import convert_to_tensor

DECELERATION_ZONES = (
    {"center": [[0.0, 0.0]], "decay": [2.0]},
    {"center": [[5.0, 4.5], [1.5, 3.0]], "decay": [1.15, 1.2]},
)


@pytest.fixture(params=DECELERATION_ZONES)
def env_config(request):
    return request.param


@pytest.fixture
def env(navigation_env, env_config):
    return navigation_env(env_config)


@pytest.fixture(params=(1, 4))
def n_batch(request):
    return request.param


@pytest.fixture(params=((), (1,), (2,)))
def sample_shape(request):
    return request.param


def test_reward_fn(env):
    obs = env.reset()
    act = env.action_space.sample()
    _obs, rew, _, _ = env.step(act)

    obs_t, act_t, _obs_t = map(
        partial(convert_to_tensor, device="cpu"), (obs, act, _obs)
    )
    rew_t = env.reward_fn(obs_t, act_t, _obs_t)

    assert np.allclose(rew, rew_t.numpy())


def test_transition_fn_fidelity(env):
    obs = env.reset()
    act = env.action_space.sample()
    torch.manual_seed(42)
    _obs, _, _, _ = env.step(act)

    obs_t, act_t = map(partial(convert_to_tensor, device="cpu"), (obs, act))
    obs_t, act_t = map(lambda x: x.requires_grad_(True), (obs_t, act_t))
    torch.manual_seed(42)
    _obs_t, logp = env.transition_fn(obs_t, act_t)

    assert _obs_t.grad_fn is not None
    assert _obs_t.detach().numpy() in env.observation_space
    assert np.allclose(_obs, _obs_t.detach().numpy())
    assert logp.shape == ()
    assert logp.dtype == torch.float32
    assert logp.grad_fn is not None


def test_transition_fn_sampling(env, n_batch, sample_shape):
    obs = [env.reset() for _ in range(n_batch)]
    act = [env.action_space.sample() for _ in range(n_batch)]
    obs_t, act_t = map(partial(convert_to_tensor, device="cpu"), (obs, act))
    obs_t, act_t = map(lambda x: x.requires_grad_(True), (obs_t, act_t))
    _obs_t, logp = env.transition_fn(obs_t, act_t, sample_shape)

    assert _obs_t.shape == sample_shape + obs_t.shape
    assert _obs_t.grad_fn is not None
    assert logp.shape == sample_shape + (n_batch,)
    assert logp.dtype == torch.float32
    assert logp.grad_fn is not None
