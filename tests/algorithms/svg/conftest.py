import pytest
import gym.spaces as spaces

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
def cartpole_swingup_env(time_limited_env):
    return lambda _: time_limited_env(
        {"env_id": "CartPoleSwingUp", "time_aware": True, "max_episode_steps": 200}
    )


@pytest.fixture(params=((1,), (2,), (4,)))
def shape(request):
    return request.param


@pytest.fixture
def obs_space(shape):
    return spaces.Box(-10, 10, shape=shape)


@pytest.fixture
def action_space(shape):
    return spaces.Box(-1, 1, shape=shape)
