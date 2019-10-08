import pytest
import numpy as np

from raylab.envs.registry import ENVS


@pytest.fixture(params=[v for k, v in ENVS.items() if k != "TimeLimitedEnv"])
def env(request):
    return request.param({})


def test_env_interaction_loop(env):
    obs = env.reset()
    assert obs in env.observation_space

    action = env.action_space.sample()
    new_obs, rew, done, info = env.step(action)
    assert new_obs in env.observation_space
    assert np.isscalar(rew)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    for _ in range(10):
        if done:
            break
        _, _, done, _ = env.step(env.action_space.sample())
