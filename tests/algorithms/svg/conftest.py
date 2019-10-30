# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest

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
def cartpole_swingup_env(envs):
    return lambda _: envs["CartPoleSwingUp"](
        {"time_aware": True, "max_episode_steps": 200}
    )


@pytest.fixture(params=range(2))
def env_creator(request, cartpole_swingup_env, navigation_env):
    creators = [cartpole_swingup_env, navigation_env]
    return creators[request.param]


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_fn, svg_policy):
    def make_policy_and_batch(config):
        return policy_and_batch_fn(svg_policy, config)

    return make_policy_and_batch
