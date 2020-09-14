import pytest
from ray.rllib import SampleBatch

from raylab.agents.registry import get_agent_cls


@pytest.fixture(params="SVG(1) SVG(inf)".split())
def svg_trainer(request):
    return get_agent_cls(request.param)


@pytest.fixture
def svg_policy(svg_trainer):
    return svg_trainer._policy


@pytest.fixture
def svg_one_trainer():
    return get_agent_cls("SVG(1)")


@pytest.fixture
def svg_one_policy(svg_one_trainer):
    return svg_one_trainer._policy


@pytest.fixture
def svg_inf_trainer():
    return get_agent_cls("SVG(inf)")


@pytest.fixture
def svg_inf_policy(svg_inf_trainer):
    return svg_inf_trainer._policy


@pytest.fixture
def batch_fn(env_samples):
    def maker(policy):
        batch = policy.lazy_tensor_dict(env_samples)
        batch[SampleBatch.REWARDS] = policy.reward_fn(
            batch[SampleBatch.CUR_OBS],
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.NEXT_OBS],
        )
        return batch

    return maker
