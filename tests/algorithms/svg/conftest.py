import pytest

from raylab.algorithms.registry import ALGORITHMS


@pytest.fixture
def svg_one_trainer():
    return ALGORITHMS["SVG(1)"]()


@pytest.fixture
def svg_one_policy(svg_one_trainer):
    return svg_one_trainer._policy


@pytest.fixture
def svg_inf_trainer():
    return ALGORITHMS["SVG(inf)"]()


@pytest.fixture
def svg_inf_policy(svg_inf_trainer):
    return svg_inf_trainer._policy


@pytest.fixture
def cartpole_swingup_env(cartpole_swingup_env):
    return lambda _: cartpole_swingup_env(
        {"time_aware": True, "max_episode_steps": 200}
    )
