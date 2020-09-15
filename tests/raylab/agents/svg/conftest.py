import pytest
from ray.rllib import SampleBatch

from raylab.agents.registry import get_agent_cls
from raylab.agents.svg.inf import SVGInfTorchPolicy
from raylab.agents.svg.one import SVGOneTorchPolicy


@pytest.fixture(params="SVG(1) SVG(inf)".split())
def svg_trainer(request):
    return get_agent_cls(request.param)


@pytest.fixture(
    params=(SVGInfTorchPolicy, SVGOneTorchPolicy), ids=lambda x: f"{x.__name__}"
)
def svg_policy(request):
    return request.param


@pytest.fixture
def svg_one_trainer():
    return get_agent_cls("SVG(1)")


@pytest.fixture
def svg_one_policy():
    return SVGOneTorchPolicy


@pytest.fixture
def svg_inf_trainer():
    return get_agent_cls("SVG(inf)")


@pytest.fixture
def svg_inf_policy():
    return SVGInfTorchPolicy


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
