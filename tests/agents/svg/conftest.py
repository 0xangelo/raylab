# pylint: disable=missing-docstring,redefined-outer-name,protected-access
import pytest
from ray.rllib import SampleBatch

from raylab.agents.registry import AGENTS

from ...mock_env import MockReward


@pytest.fixture(
    params=[AGENTS[k] for k in "SVG(1) SVG(inf)".split()], ids=("SVG(1)", "SVF(inf)")
)
def svg_trainer(request):
    return request.param()


@pytest.fixture
def svg_policy(svg_trainer):
    return svg_trainer._policy


@pytest.fixture
def svg_one_trainer():
    return AGENTS["SVG(1)"]()


@pytest.fixture
def svg_one_policy(svg_one_trainer):
    return svg_one_trainer._policy


@pytest.fixture
def svg_inf_trainer():
    return AGENTS["SVG(inf)"]()


@pytest.fixture
def svg_inf_policy(svg_inf_trainer):
    return svg_inf_trainer._policy


@pytest.fixture
def policy_and_batch_fn(policy_and_batch_):
    def make_policy_and_batch(policy_cls, config):
        config["env"] = "MockEnv"
        policy, batch = policy_and_batch_(policy_cls, config)
        reward_fn = MockReward({})
        batch[SampleBatch.REWARDS] = reward_fn(
            batch[SampleBatch.CUR_OBS],
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.NEXT_OBS],
        )
        return policy, batch

    return make_policy_and_batch
