# pylint: disable=missing-docstring,redefined-outer-name
from functools import partial

import pytest
import numpy as np

from raylab.utils.pytorch import convert_to_tensor
from raylab.envs.registry import ENVS


@pytest.fixture
def env():
    env = ENVS["CartPoleSwingUp"]({})
    return env


def test_reward_fn(env):
    obs = env.reset()
    act = env.action_space.sample()
    _obs, rew, _, _ = env.step(act)

    obs_t, act_t, _obs_t = map(
        partial(convert_to_tensor, device="cpu"), (obs, act, _obs)
    )
    rew_t = env.reward_fn(obs_t, act_t, _obs_t)

    assert np.allclose(rew, rew_t.numpy(), atol=1e-6)
