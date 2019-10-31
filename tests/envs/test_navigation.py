# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import numpy as np
import torch

from raylab.utils.pytorch import convert_to_tensor


@pytest.fixture
def env(navigation_env):
    return navigation_env({})


def test_reward_fn(env):
    obs = env.reset()
    act = env.action_space.sample()
    _obs, rew, _, _ = env.step(act)

    obs_t, act_t, _obs_t = map(
        partial(convert_to_tensor, device="cpu"), (obs, act, _obs)
    )
    rew_t = env.reward_fn(obs_t, act_t, _obs_t)

    assert np.allclose(rew, rew_t.numpy())


def test_transition_fn(env):
    obs = env.reset()
    act = env.action_space.sample()
    torch.manual_seed(42)
    _obs, _, _, _ = env.step(act)

    obs_t, act_t = map(partial(convert_to_tensor, device="cpu"), (obs, act))
    obs_t, act_t = map(lambda x: x.requires_grad_(True), (obs_t, act_t))
    torch.manual_seed(42)
    _obs_t, logp = env.transition_fn(obs_t, act_t)

    assert _obs_t.grad_fn is None
    assert np.allclose(_obs, _obs_t.numpy())
    assert logp.shape == ()
    assert logp.dtype == torch.float32
    assert logp.grad_fn is not None
