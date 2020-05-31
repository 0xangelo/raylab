# pylint: disable=missing-docstring,redefined-outer-name,protected-access
from functools import partial

import pytest
import torch

from raylab.utils.pytorch import convert_to_tensor

try:
    import gym_cartpole_swingup  # pylint:disable=unused-import
except ImportError:
    pytest.skip(
        "Missing gym-cartpole-swingup; skipping tests...", allow_module_level=True
    )


@pytest.fixture(params="TorchCartPoleSwingUp-v0 TorchCartPoleSwingUp-v1".split())
def env(request, envs):
    return envs[request.param]({})


@pytest.fixture(params=((), (1,), (4,)))
def sample_shape(request):
    return request.param


def test_transition_fn(env, sample_shape):
    obs = env.reset()
    act = env.action_space.sample()
    _obs, _, _, _ = env.step(act)

    obs_t, act_t, _obs_t = map(
        partial(convert_to_tensor, device="cpu"), (obs[None], act[None], _obs[None])
    )
    samp, _ = env.transition_fn(obs_t, act_t, sample_shape=sample_shape)

    assert samp.shape == sample_shape + obs_t.shape
    assert torch.allclose(_obs_t.expand_as(samp), samp, atol=1e-5)
