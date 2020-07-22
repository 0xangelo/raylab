import numpy as np
import pytest
import torch

from raylab.envs import get_env_creator


@pytest.fixture
def env(env_name):
    return get_env_creator(env_name)({})


def test_env_interaction_loop(env):
    obs = env.reset()
    assert obs in env.observation_space

    action = env.action_space.sample()
    new_obs, rew, done, info = env.step(action)
    assert new_obs in env.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    if hasattr(env, "reward_fn"):
        rewt = env.reward_fn(*map(torch.Tensor, (obs, action, new_obs)))
        assert torch.allclose(torch.as_tensor(rew), rewt)

    for _ in range(10):
        if done:
            break
        _, _, done, _ = env.step(env.action_space.sample())
